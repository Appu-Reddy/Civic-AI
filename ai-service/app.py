from flask import Flask, request, jsonify
import logging
from pathlib import Path
from typing import Optional

from agents.graph import run_marag
from ingestion import parser as parser_module
from ingestion.chunker import chunk_pages
from ingestion.embedder import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MODEL_NAME,
    OUT_DIR as DEFAULT_CHUNKS_DIR,
    PDF_DIR as DEFAULT_PDF_DIR,
    embed_chunks,
    save_embeddings,
)
from vectorDB.faiss_util import CHUNKS_DIR as DEFAULT_FAISS_CHUNKS_DIR, FAISS_DIR as DEFAULT_FAISS_DIR, build_index

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

LOGGER = logging.getLogger(__name__)


def _resolve_path(value: Optional[str], default_path: Path) -> str:
    if value:
        return str(Path(value))
    return str(default_path)


@app.route("/api/v1/embed", methods=["POST"])
def build_embeddings():
    data = request.get_json(silent=True) or {}

    pdf_dir = _resolve_path(data.get("pdf_dir"), DEFAULT_PDF_DIR)
    out_dir = _resolve_path(data.get("out_dir"), DEFAULT_CHUNKS_DIR)
    chunk_size = int(data.get("chunk_size", CHUNK_SIZE))
    chunk_overlap = int(data.get("chunk_overlap", CHUNK_OVERLAP))
    batch_size = int(data.get("batch_size", BATCH_SIZE))

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        return jsonify({
            "success": False,
            "message": f"PDF directory not found: {pdf_dir}"
        }), 400

    try:
        pages = parser_module.parse_directory(pdf_dir)
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings, metadata = embed_chunks(chunks, model_name=MODEL_NAME, batch_size=batch_size)
        save_embeddings(embeddings, metadata, out_dir)

        summary = {
            "total_files": parser_module.LAST_TOTAL_FILES,
            "total_pages": parser_module.LAST_TOTAL_PAGES,
            "total_chunks": len(metadata),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
            "failed_files": sorted(set(parser_module.LAST_FAILED_FILES)),
        }

        return jsonify({
            "success": True,
            "message": "Parser, chunker, and embedder completed successfully.",
            "data": summary
        })

    except Exception as exc:
        LOGGER.exception("Embedding pipeline failed")

        return jsonify({
            "success": False,
            "message": str(exc)
        }), 500


@app.route("/api/v1/faiss", methods=["POST"])
def build_faiss_index():
    data = request.get_json(silent=True) or {}

    chunks_dir = _resolve_path(data.get("chunks_dir"), DEFAULT_FAISS_CHUNKS_DIR)
    out_dir = _resolve_path(data.get("out_dir"), DEFAULT_FAISS_DIR)

    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        return jsonify({
            "success": False,
            "message": f"Chunks directory not found: {chunks_dir}"
        }), 400

    try:
        summary = build_index(chunks_dir=chunks_dir, out_dir=out_dir)

        return jsonify({
            "success": True,
            "message": "FAISS index generated successfully.",
            "data": summary
        })

    except Exception as exc:
        LOGGER.exception("FAISS index build failed")

        return jsonify({
            "success": False,
            "message": str(exc)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/api/v1/query", methods=["POST"])
def query():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "Missing JSON body."
        }), 400

    query = data.get("query")

    if not query:
        return jsonify({
            "success": False,
            "message": "Field 'query' is required."
        }), 400

    top_k = data.get("top_k", 5)
    max_steps = data.get("max_steps")

    try:
        kwargs = {
            "query": query,
            "top_k": top_k,
        }

        if max_steps is not None:
            kwargs["max_steps"] = max_steps

        result = run_marag(**kwargs)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        LOGGER.exception("Query execution failed")

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/api/v1/query/answer", methods=["POST"])
def answer_only():

    data = request.get_json(silent=True)

    if not data or "query" not in data:
        return jsonify({
            "success": False,
            "message": "Field 'query' is required."
        }), 400

    try:
        result = run_marag(
            query=data["query"],
            top_k=data.get("top_k", 5),
            max_steps=data.get("max_steps", 10)
        )

        return jsonify({
            "success": True,
            "query": result["query"],
            "answer": result["final_answer"]
        })

    except Exception as e:
        LOGGER.exception("Answer generation failed")

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=3010,
        debug=True
    )