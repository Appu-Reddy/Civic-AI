"""
app.py — Civic-AI Flask API

Endpoints
---------
GET  /api/v1/health             Index + infrastructure readiness check
POST /api/v1/embed              Synchronous full re-ingestion (admin / CI use)
POST /api/v1/upload             Async PDF upload → RabbitMQ (falls back to sync)
GET  /api/v1/status/<job_id>    Poll async ingestion job status (Redis-backed)
POST /api/v1/query              Natural-language Q&A with Redis query caching

Caching strategy
----------------
Query results are cached in Redis keyed by a SHA-256 of the normalised query
string.  On a cache hit the pipeline is skipped entirely and the stored dict
is returned directly.  Caches are flushed whenever indexes are rebuilt so
answers never go stale.

Async ingestion
---------------
POST /api/v1/upload publishes a job to RabbitMQ ("docqa.ingest" queue) and
returns immediately with a job_id.  If RabbitMQ is unreachable the endpoint
falls back to synchronous ingestion so the API stays usable without the broker.
The worker process (worker.py) consumes the queue, runs run_ingestion(), and
writes the result back to Redis so /status/<job_id> can report progress.
"""

import logging
import threading
from pathlib import Path

for _logger_name in (
    "pika",
    "redis",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from ai.pipeline import load_retrieval_context, run_pipeline
from database.mongodb import get_collection
from ingestion.ingest import run_ingestion
from services.redis import (
    flush_all_caches,
    get_job_status,
    get_query_cache,
    set_query_cache,
)
from services.rabbitmq import publish_ingestion_job

BASE_DIR      = Path(__file__).resolve().parent
INDEX_DIR     = BASE_DIR / "data" / "indexes" / "faiss"
GRAPH_DIR     = BASE_DIR / "data" / "indexes" / "graph"
PDF_DIR       = BASE_DIR / "data" / "pdfs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

app = Flask(__name__)

# ── In-process retrieval context (FAISS + graph + embedder) ──────────────────
_context      = None
_context_lock = threading.Lock()


def _load_context():
    """Load indexes and the embedding model once, on the first query."""
    global _context
    if _context is not None:
        return _context

    with _context_lock:
        if _context is not None:
            return _context

        mongo_collection = None
        try:
            mongo_collection = get_collection()
        except Exception:
            pass

        _context = load_retrieval_context(
            index_dir=INDEX_DIR,
            graph_dir=GRAPH_DIR,
            mongo_collection=mongo_collection,
        )
        return _context


def _reset_context() -> None:
    """Invalidate the in-process context so the next query reloads the indexes."""
    global _context
    with _context_lock:
        _context = None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    """Return readiness of indexes and optional infrastructure services."""
    faiss_ready    = (INDEX_DIR / "civic_ai.index").exists()
    metadata_ready = (INDEX_DIR / "chunk_metadata_map.pkl").exists()
    graph_ready    = (GRAPH_DIR / "civic_ai_graph.pkl").exists()

    mongo_ready = False
    try:
        get_collection().database.client.admin.command("ping")
        mongo_ready = True
    except Exception:
        pass

    from services.redis import is_available as redis_up
    from services.rabbitmq import is_available as rabbit_up
    redis_ready  = redis_up()
    rabbit_ready = rabbit_up()

    ready = faiss_ready and metadata_ready and graph_ready
    return jsonify({
        "status": "ok" if ready else "not_ready",
        "indexes": {
            "faiss":         faiss_ready,
            "faiss_metadata": metadata_ready,
            "graph":          graph_ready,
        },
        "services": {
            "mongodb":  mongo_ready,
            "redis":    redis_ready,
            "rabbitmq": rabbit_ready,
        },
    }), (200 if ready else 503)


# ── Embed (synchronous full re-ingestion) ─────────────────────────────────────

@app.post("/api/v1/embed")
def embed():
    """
    Trigger a full synchronous ingestion of all PDFs in data/pdfs/.
    Intended for admin / CI use where async is not needed.

    Flushes the Redis cache after a successful run so no stale answers remain.
    """
    payload = request.get_json(silent=True) or {}
    rebuild = bool(payload.get("rebuild", False))

    try:
        summary = run_ingestion(
            index_dir=INDEX_DIR,
            graph_dir=GRAPH_DIR,
            rebuild=rebuild,
        )
        _reset_context()
        cache_summary = flush_all_caches()
        return jsonify({
            "status":        "completed",
            "summary":       summary,
            "cache_flushed": cache_summary,
        }), 200
    except Exception as exc:
        logger.exception("Embedding pipeline failed")
        return jsonify({"status": "failed", "error": str(exc)}), 500


# ── Upload (async via RabbitMQ, sync fallback) ────────────────────────────────

@app.post("/api/v1/upload")
def upload():
    """
    Upload a PDF and enqueue it for ingestion.

    Flow (RabbitMQ available):
        1. Validate and save the PDF to data/pdfs/
        2. Publish a job message to RabbitMQ
        3. Return 202 Accepted with job_id for status polling

    Flow (RabbitMQ unavailable — fallback):
        1. Validate and save the PDF to data/pdfs/
        2. Run ingestion synchronously
        3. Return 200 with ingestion summary

    Returns 409 if the file already exists and rebuild was not requested.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use field name 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename provided."}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 415

    rebuild = request.form.get("rebuild", "false").lower() == "true"

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDF_DIR / filename

    if dest.exists() and not rebuild:
        return jsonify({
            "error": f"'{filename}' already exists. Send rebuild=true to overwrite and re-index.",
        }), 409

    file.save(dest)

    # ── Try async path first ──────────────────────────────────────────────────
    try:
        job_id = publish_ingestion_job(filename=filename, rebuild=rebuild)
        return jsonify({
            "status":  "queued",
            "async":   True,
            "job_id":  job_id,
            "filename": filename,
            "message": (
                f"Ingestion queued. Poll GET /api/v1/status/{job_id} for progress."
            ),
        }), 202

    except ConnectionError as broker_err:
        logger.warning(
            "RabbitMQ unavailable (%s) — falling back to synchronous ingestion.",
            broker_err,
        )

    # ── Sync fallback ─────────────────────────────────────────────────────────
    try:
        summary = run_ingestion(
            pdf_dir=PDF_DIR,
            index_dir=INDEX_DIR,
            graph_dir=GRAPH_DIR,
            processed_dir=PROCESSED_DIR,
            rebuild=rebuild,
        )
        _reset_context()
        cache_summary = flush_all_caches()
        return jsonify({
            "status":        "completed",
            "async":         False,
            "filename":      filename,
            "summary":       summary,
            "cache_flushed": cache_summary,
        }), 200

    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Upload ingestion failed")
        return jsonify({"status": "failed", "filename": filename, "error": str(exc)}), 500


# ── Job status ────────────────────────────────────────────────────────────────

@app.get("/api/v1/status/<job_id>")
def job_status(job_id: str):
    """
    Return the current status of an async ingestion job.

    Status values: queued | processing | completed | failed

    Returns 404 if the job_id is unknown (job never existed, or TTL expired).
    Returns 503 if Redis is unavailable (cannot look up status).
    """
    from services.redis import is_available as redis_up
    if not redis_up():
        return jsonify({
            "error": "Status store (Redis) is unavailable. Cannot look up job status.",
        }), 503

    status = get_job_status(job_id)
    if status is None:
        return jsonify({
            "error": f"Job '{job_id}' not found. It may have expired or never existed.",
        }), 404

    # If the job completed, also reset the in-process retrieval context so the
    # next query picks up the freshly built indexes from this API process too.
    if status.get("status") == "completed":
        _reset_context()

    http_code = 200
    if status.get("status") == "failed":
        http_code = 500

    return jsonify(status), http_code


# ── Query (with Redis cache) ──────────────────────────────────────────────────

@app.post("/api/v1/query")
def query():
    """
    Answer a natural-language question using the AI pipeline.

    Cache behaviour
    ---------------
    1. Normalise the query string and check Redis.
    2. Cache hit  → return the stored response immediately (no pipeline call).
    3. Cache miss → run the full pipeline, store the result, return it.

    The 'no_cache' field in the JSON body (bool, default false) bypasses the
    cache for this request without invalidating existing entries — useful for
    debugging or forcing a fresh answer.
    """
    payload    = request.get_json(silent=True) or {}
    query_text = payload.get("query")
    no_cache   = bool(payload.get("no_cache", False))

    if not isinstance(query_text, str) or not query_text.strip():
        return jsonify({"error": "JSON field 'query' must be a non-empty string."}), 400

    query_text = query_text.strip()

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if not no_cache:
        cached = get_query_cache(query_text)
        if cached is not None:
            cached["cached"] = True
            return jsonify(cached), 200

    # ── Cache miss: run pipeline ──────────────────────────────────────────────
    try:
        context  = _load_context()
        response = run_pipeline(
            query=query_text,
            retrieval_ctx=context,
            final_top_k=int(payload.get("final_top_k", 5)),
        )
        result = response.to_dict()
        result["cached"] = False

        # Store in cache (no-ops silently if Redis is unavailable)
        if not no_cache:
            set_query_cache(query_text, result)

        return jsonify(result), 200

    except FileNotFoundError as exc:
        return jsonify({
            "error":  "Indexes are not ready. Call POST /api/v1/embed first.",
            "detail": str(exc),
        }), 503
    except Exception as exc:
        logger.exception("Query pipeline failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)
