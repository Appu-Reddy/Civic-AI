"""
rabbitmq.py — RabbitMQ publisher (API side) and consumer (worker side).

Architecture
------------
API process  →  publish_ingestion_job()  →  RabbitMQ queue "docqa.ingest"
Worker process  →  start_worker()  →  consumes "docqa.ingest"  →  run_ingestion()

Message schema (JSON)
---------------------
{
    "job_id":   "<uuid4>",          # unique identifier for status polling
    "filename": "report.pdf",       # filename already saved to data/pdfs/
    "rebuild":  false               # passed through to run_ingestion()
}

Job status lifecycle stored in Redis
-------------------------------------
queued  →  processing  →  completed
                       →  failed
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUEUE_NAME: str  = "docqa.ingest"
RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


# ── Connection helper ─────────────────────────────────────────────────────────

def _get_connection():
    """
    Open and return a blocking pika connection.
    Raises ConnectionError (wrapping the original exception) if unreachable.
    """
    try:
        import pika
        params = pika.URLParameters(RABBITMQ_URL)
        params.socket_timeout = 5
        logger.info("INFO: Connected to Rabbit-MQ.")
        return pika.BlockingConnection(params)
    except Exception as exc:
        raise ConnectionError(f"RabbitMQ unavailable at {RABBITMQ_URL}: {exc}") from exc


def is_available() -> bool:
    """Return True if RabbitMQ can be reached right now."""
    try:
        conn = _get_connection()
        conn.close()
        return True
    except Exception:
        return False


# ── Publisher (API process) ───────────────────────────────────────────────────

def publish_ingestion_job(filename: str, rebuild: bool = False) -> str:
    """
    Enqueue an ingestion job for *filename* and return the job_id.

    The PDF must already exist at data/pdfs/<filename> before calling this.
    Immediately sets job status to "queued" in Redis (if available).

    Raises ConnectionError if RabbitMQ is unreachable.
    """
    import pika

    job_id = str(uuid.uuid4())
    message = json.dumps({
        "job_id":   job_id,
        "filename": filename,
        "rebuild":  rebuild,
    })

    conn = _get_connection()
    try:
        channel = conn.channel()
        channel.queue_declare(
            queue=QUEUE_NAME,
            durable=True,   # survive RabbitMQ restart
        )
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent,  # survive broker restart
                content_type="application/json",
            ),
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Mark the job as queued in Redis so /status/<job_id> can respond immediately
    try:
        from .redis import set_job_status
        set_job_status(job_id, {
            "job_id":   job_id,
            "filename": filename,
            "status":   "queued",
            "rebuild":  rebuild,
            "summary":  None,
            "error":    None,
        })
    except Exception as exc:
        logger.debug("Could not persist initial job status to Redis: %s", exc)

    logger.info("queued: job_id=%s filename=%s", job_id, filename)
    return job_id


# ── Consumer (worker process) ─────────────────────────────────────────────────

def _process_message(body: bytes) -> None:
    """
    Handle one ingestion message.  Called by the worker's consume loop.

    Lifecycle:
        1. Decode JSON message
        2. Set status → "processing" in Redis
        3. Run run_ingestion() for the named file
        4. Reset in-process retrieval context (if running in-process)
        5. Flush Redis caches so stale query results are evicted
        6. Set status → "completed" or "failed" in Redis
    """
    from .redis import set_job_status, flush_all_caches

    try:
        payload = json.loads(body)
    except Exception as exc:
        logger.error("Worker: failed to decode message: %s — body: %r", exc, body)
        return

    job_id   = payload.get("job_id", "unknown")
    filename = payload.get("filename", "")
    rebuild  = bool(payload.get("rebuild", False))

    processing_status = {
        "job_id":   job_id,
        "filename": filename,
        "status":   "processing",
        "rebuild":  rebuild,
        "summary":  None,
        "error":    None,
    }
    set_job_status(job_id, processing_status)
    logger.info("working: job_id=%s filename=%s", job_id, filename)
    try:
        from database.query_store import update_upload_status
        update_upload_status(job_id, "processing")
    except Exception:
        pass

    try:
        from pathlib import Path as _Path
        BASE_DIR      = _Path(__file__).resolve().parent.parent
        INDEX_DIR     = BASE_DIR / "data" / "indexes" / "faiss"
        GRAPH_DIR     = BASE_DIR / "data" / "indexes" / "graph"
        PDF_DIR       = BASE_DIR / "data" / "pdfs"
        PROCESSED_DIR = BASE_DIR / "data" / "processed"

        from ingestion.ingest import run_ingestion
        summary = run_ingestion(
            pdf_dir=PDF_DIR,
            index_dir=INDEX_DIR,
            graph_dir=GRAPH_DIR,
            processed_dir=PROCESSED_DIR,
            rebuild=rebuild,
        )

        # Flush both caches — indexes have changed, all cached answers are stale
        flush_all_caches()

        completed_status = {
            "job_id":   job_id,
            "filename": filename,
            "status":   "completed",
            "rebuild":  rebuild,
            "summary":  summary,
            "error":    None,
        }
        set_job_status(job_id, completed_status)
        try:
            from database.query_store import update_upload_status
            update_upload_status(job_id, "completed", summary=summary)
        except Exception:
            pass
        logger.info("done: job_id=%s filename=%s", job_id, filename)

    except Exception as exc:
        logger.exception("Worker failed: job_id=%s", job_id)
        failed_status = {
            "job_id":   job_id,
            "filename": filename,
            "status":   "failed",
            "rebuild":  rebuild,
            "summary":  None,
            "error":    str(exc),
        }
        set_job_status(job_id, failed_status)
        try:
            from database.query_store import update_upload_status
            update_upload_status(job_id, "failed", error=str(exc))
        except Exception:
            pass


def start_worker(
    prefetch_count: int = 1,
    on_message: Optional[Callable[[bytes], None]] = None,
) -> None:
    """
    Start a blocking consumer loop on QUEUE_NAME.

    Parameters
    ----------
    prefetch_count : int
        How many unacknowledged messages the worker holds at once.
        Default 1 ensures jobs are processed one at a time (safe for
        the heavy ingestion pipeline).
    on_message : callable, optional
        Override the default _process_message handler — useful for testing.

    This function blocks forever (or until the connection drops / KeyboardInterrupt).
    Run it in a dedicated process (see worker.py).
    """
    import pika

    handler = on_message or _process_message

    while True:
        try:
            conn = _get_connection()
            channel = conn.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=prefetch_count)

            def _callback(ch, method, properties, body):
                try:
                    handler(body)
                except Exception as exc:
                    logger.exception("Worker: unhandled error in message handler: %s", exc)
                finally:
                    ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_callback)
            logger.debug("Worker ready — consuming queue '%s'", QUEUE_NAME)
            channel.start_consuming()

        except KeyboardInterrupt:
            break
        except Exception as exc:
            import time
            logger.error("Worker connection lost (%s) — retrying in 5 s...", exc)
            time.sleep(5)
