"""
worker.py — RabbitMQ ingestion worker for DocQA.

Run this as a separate process alongside the Flask API:

    python -m services.worker

The worker:
  1. Connects to RabbitMQ (RABBITMQ_URL from .env).
  2. Listens on the "docqa.ingest" queue.
  3. For each message: parses the PDF filename, runs the full ingestion
     pipeline (parse → chunk → embed → FAISS + graph + MongoDB), flushes
     the Redis query/retrieval cache, and writes the final job status back
     to Redis so the API's /status/<job_id> endpoint can report it.
  4. Reconnects automatically if the broker connection drops.

One worker process handles one job at a time (prefetch_count=1) to avoid
overwhelming the system with concurrent heavy ingestion jobs.  Start multiple
worker processes if parallel throughput is needed.

Environment variables (all optional — defaults shown):
    RABBITMQ_URL    amqp://guest:guest@localhost:5672/
    REDIS_URL       redis://localhost:6379
"""

import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SERVICES_DIR = Path(__file__).resolve().parent
while str(SERVICES_DIR) in sys.path:
    sys.path.remove(str(SERVICES_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.WARNING)
logging.getLogger("services.redis").setLevel(logging.ERROR)
logging.getLogger("services.rabbitmq").setLevel(logging.INFO)
logger = logging.getLogger("docqa.worker")
logger.setLevel(logging.INFO)


def main() -> None:
    from services.rabbitmq import start_worker

    try:
        start_worker(prefetch_count=1)
    except KeyboardInterrupt:
        logger.info("Worker stopped.")
    except Exception as exc:
        logger.critical("Worker exited with unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
