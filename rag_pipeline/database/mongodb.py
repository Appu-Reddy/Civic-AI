import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, ConnectionFailure

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME: str = "CivicAI"
COLLECTION_NAME: str = "embeddings"


def get_collection(
    uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = COLLECTION_NAME,
) -> Collection:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

    collection = client[db_name][collection_name]
    collection.create_index([("document_id", ASCENDING)], background=True)
    collection.create_index([("domain", ASCENDING)], background=True)
    collection.create_index([("page_number", ASCENDING)], background=True)
    return collection


def _embedded_chunk_to_doc(embedded_chunk) -> dict:
    return {
        "_id": embedded_chunk.chunk_id,
        "document_id": embedded_chunk.document_id,
        "document_name": embedded_chunk.document_name,
        "page_number": embedded_chunk.page_number,
        "section": embedded_chunk.section,
        "text": embedded_chunk.text,
        "word_count": embedded_chunk.word_count,
        "char_offset": embedded_chunk.char_offset,
        "domain": embedded_chunk.domain,
        "source": embedded_chunk.source,
        "embedding": embedded_chunk.embedding.tolist(),
        "embedding_model": embedded_chunk.embedding_model,
    }


def store_embedded_chunks(embedded_chunks: list, collection: Collection) -> int:
    if not embedded_chunks:
        return 0

    docs = [_embedded_chunk_to_doc(ec) for ec in embedded_chunks]
    try:
        result = collection.insert_many(docs, ordered=False)
        return len(result.inserted_ids)
    except BulkWriteError as bwe:
        inserted = bwe.details.get("nInserted", 0)
        dup_errors = [e for e in bwe.details.get("writeErrors", []) if e.get("code") == 11000]
        other_errors = len(bwe.details.get("writeErrors", [])) - len(dup_errors)
        if other_errors > 0:
            return inserted


def get_chunk_by_id(chunk_id: str, collection: Collection) -> Optional[dict]:
    return collection.find_one({"_id": chunk_id})


def get_chunks_by_document(document_id: str, collection: Collection, limit: int = 0) -> list[dict]:
    cursor = collection.find({"document_id": document_id}).sort([("page_number", ASCENDING), ("char_offset", ASCENDING)])
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def get_chunks_by_domain(domain: str, collection: Collection, limit: int = 100) -> list[dict]:
    return list(collection.find({"domain": domain}).sort("page_number", ASCENDING).limit(limit))


def get_chunk_ids_for_document(document_id: str, collection: Collection) -> list[str]:
    return [doc["_id"] for doc in collection.find({"document_id": document_id}, {"_id": 1})]


def get_embeddings_array(collection: Collection, document_id: Optional[str] = None) -> tuple[np.ndarray, list[str]]:
    query = {"document_id": document_id} if document_id else {}
    docs = list(collection.find(query, {"_id": 1, "embedding": 1}))
    if not docs:
        return np.empty((0,), dtype=np.float32), []
    return np.array([d["embedding"] for d in docs], dtype=np.float32), [d["_id"] for d in docs]


def list_documents(collection: Collection) -> list[dict]:
    pipeline = [
        {"$group": {"_id": "$document_id", "document_name": {"$first": "$document_name"},
                    "source": {"$first": "$source"}, "chunk_count": {"$sum": 1}, "domains": {"$addToSet": "$domain"}}},
        {"$sort": {"_id": ASCENDING}},
    ]
    return list(collection.aggregate(pipeline))


def clear_collection(collection: Collection) -> int:
    result = collection.delete_many({})
    return result.deleted_count