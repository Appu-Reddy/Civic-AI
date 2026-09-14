"""
query_store.py — Persist query history and upload jobs in MongoDB.

Database : Civic-AI  (user-specified, separate from CivicAI embeddings db)
Collections:
    query_data   — saved Q&A responses from /api/v1/query
    upload_data  — PDF upload jobs with RabbitMQ job IDs
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from pathlib import Path
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
APP_DB_NAME: str = os.getenv("APP_DB_NAME", "CivicAI")
QUERY_COLLECTION: str = "query_data"
UPLOAD_COLLECTION: str = "upload_data"

_client = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        try:
            _client.admin.command("ping")
        except ConnectionFailure as exc:
            logger.error("MongoDB connection failed: %s", exc)
            raise
    return _client


def get_query_collection() -> Collection:
    col = _get_client()[APP_DB_NAME][QUERY_COLLECTION]
    col.create_index([("created_at", DESCENDING)], background=True)
    return col


def get_upload_collection() -> Collection:
    col = _get_client()[APP_DB_NAME][UPLOAD_COLLECTION]
    col.create_index([("created_at", DESCENDING)], background=True)
    col.create_index([("job_id", ASCENDING)], background=True, unique=True)
    return col


def save_query_record(result: dict) -> Optional[str]:
    """Persist a pipeline response dict. Returns the inserted document id."""
    try:
        col = get_query_collection()
        doc_id = str(uuid.uuid4())
        doc = {
            "_id": doc_id,
            "query": result.get("query", ""),
            "answer": result.get("answer", ""),
            "is_multi_hop": bool(result.get("is_multi_hop", False)),
            "is_sufficient": bool(result.get("is_sufficient", True)),
            "is_valid": bool(result.get("is_valid", True)),
            "grounding_score": float(result.get("grounding_score", 0.0)),
            "quality_score": float(result.get("quality_score", 0.0)),
            "citations": result.get("citations", []),
            "steps": result.get("steps", []),
            "flagged_sentences": result.get("flagged_sentences", []),
            "validation_note": result.get("validation_note", ""),
            "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
            "cached": bool(result.get("cached", False)),
            "created_at": _utcnow(),
        }
        col.insert_one(doc)
        return doc_id
    except Exception as exc:
        logger.warning("save_query_record failed: %s", exc)
        return None


def list_queries(limit: int = 50, skip: int = 0) -> list[dict]:
    try:
        col = get_query_collection()
        cursor = col.find({}, {"_id": 1, "query": 1, "answer": 1, "is_multi_hop": 1,
                               "grounding_score": 1, "is_valid": 1, "citations": 1,
                               "elapsed_seconds": 1, "cached": 1, "created_at": 1})
        cursor = cursor.sort("created_at", DESCENDING).skip(skip).limit(limit)
        return [_serialize(doc) for doc in cursor]
    except Exception as exc:
        logger.warning("list_queries failed: %s", exc)
        return []


def get_query_by_id(query_id: str) -> Optional[dict]:
    try:
        col = get_query_collection()
        doc = col.find_one({"_id": query_id})
        return _serialize(doc) if doc else None
    except Exception as exc:
        logger.warning("get_query_by_id failed: %s", exc)
        return None


def query_stats() -> dict:
    try:
        col = get_query_collection()
        total = col.count_documents({})
        multi_hop = col.count_documents({"is_multi_hop": True})
        pipeline = [
            {"$group": {"_id": None, "avg_grounding": {"$avg": "$grounding_score"}}},
        ]
        agg = list(col.aggregate(pipeline))
        avg_grounding = round(agg[0]["avg_grounding"], 3) if agg else 0.0
        return {"total": total, "multi_hop": multi_hop, "avg_grounding": avg_grounding}
    except Exception as exc:
        logger.warning("query_stats failed: %s", exc)
        return {"total": 0, "multi_hop": 0, "avg_grounding": 0.0}


def save_upload_record(
    job_id: str,
    filename: str,
    status: str,
    async_mode: bool,
    rebuild: bool = False,
    summary: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    try:
        col = get_upload_collection()
        now = _utcnow()
        existing = col.find_one({"job_id": job_id})
        payload: dict[str, Any] = {
            "job_id": job_id,
            "filename": filename,
            "status": status,
            "async": async_mode,
            "rebuild": rebuild,
            "summary": summary,
            "error": error,
            "updated_at": now,
        }
        if existing:
            col.update_one({"job_id": job_id}, {"$set": payload})
        else:
            payload["created_at"] = now
            col.insert_one(payload)
    except Exception as exc:
        logger.warning("save_upload_record failed: %s", exc)


def update_upload_status(
    job_id: str,
    status: str,
    summary: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    try:
        col = get_upload_collection()
        update: dict[str, Any] = {"status": status, "updated_at": _utcnow()}
        if summary is not None:
            update["summary"] = summary
        if error is not None:
            update["error"] = error
        col.update_one({"job_id": job_id}, {"$set": update})
    except Exception as exc:
        logger.warning("update_upload_status failed: %s", exc)


def get_upload_by_job_id(job_id: str) -> Optional[dict]:
    try:
        col = get_upload_collection()
        return _serialize(col.find_one({"job_id": job_id}))
    except Exception as exc:
        logger.warning("get_upload_by_job_id failed: %s", exc)
        return None


def list_uploads(limit: int = 50, skip: int = 0) -> list[dict]:
    try:
        col = get_upload_collection()
        cursor = col.find({}).sort("created_at", DESCENDING).skip(skip).limit(limit)
        return [_serialize(doc) for doc in cursor]
    except Exception as exc:
        logger.warning("list_uploads failed: %s", exc)
        return []


def upload_stats() -> dict:
    try:
        col = get_upload_collection()
        total = col.count_documents({})
        completed = col.count_documents({"status": "completed"})
        processing = col.count_documents({"status": {"$in": ["queued", "processing"]}})
        failed = col.count_documents({"status": "failed"})
        return {
            "total": total,
            "completed": completed,
            "processing": processing,
            "failed": failed,
        }
    except Exception as exc:
        logger.warning("upload_stats failed: %s", exc)
        return {"total": 0, "completed": 0, "processing": 0, "failed": 0}


def _serialize(doc: Optional[dict]) -> Optional[dict]:
    if doc is None:
        return None
    out = dict(doc)
    if "_id" in out and out["_id"] != out.get("job_id"):
        out["id"] = str(out.pop("_id"))
    elif "_id" in out:
        out["id"] = str(out["_id"])
        del out["_id"]
    for key in ("created_at", "updated_at"):
        if key in out and hasattr(out[key], "isoformat"):
            out[key] = out[key].isoformat()
    return out
