import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR: Path = Path(os.getenv("FAISS_INDEX_DIR", str(BASE_DIR / "data" / "indexes" / "faiss")))
INDEX_FILE = "civic_ai.index"
MAPPING_FILE = "chunk_id_map.pkl"
METADATA_FILE = "chunk_metadata_map.pkl"


class FaissSearchResult:
    __slots__ = ("chunk_id", "score", "rank", "metadata")

    def __init__(self, chunk_id: str, score: float, rank: int, metadata: Optional[dict] = None) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.rank = rank
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        doc = self.metadata.get("document_name", "")
        page = self.metadata.get("page_number", "?")
        return f"FaissSearchResult(rank={self.rank}, score={self.score:.4f}, doc='{doc}' p{page})"


def build_index(embedded_chunks: list, index_dir: Path = DEFAULT_INDEX_DIR) -> faiss.Index:
    if not embedded_chunks:
        raise ValueError("Cannot build FAISS index from empty chunk list.")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embedded_chunks[0].embedding.shape[0]
    matrix = np.array([ec.embedding for ec in embedded_chunks], dtype=np.float32)
    faiss.normalize_L2(matrix)

    chunk_ids: list[str] = [ec.chunk_id for ec in embedded_chunks]
    chunk_metadata: dict[str, dict] = {
        ec.chunk_id: {
            "document_id": ec.document_id, "document_name": ec.document_name,
            "page_number": ec.page_number, "section": ec.section, "text": ec.text,
            "word_count": ec.word_count, "domain": ec.domain, "source": ec.source,
            "embedding_model": ec.embedding_model,
        }
        for ec in embedded_chunks
    }

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, str(index_dir / INDEX_FILE))
    with open(index_dir / MAPPING_FILE, "wb") as f:
        pickle.dump(chunk_ids, f)
    with open(index_dir / METADATA_FILE, "wb") as f:
        pickle.dump(chunk_metadata, f)

    return index


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> tuple[faiss.Index, list[str], dict[str, dict]]:
    index_dir = Path(index_dir)
    index_path = index_dir / INDEX_FILE
    mapping_path = index_dir / MAPPING_FILE
    metadata_path = index_dir / METADATA_FILE

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Chunk ID map not found: {mapping_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Chunk metadata map not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    with open(mapping_path, "rb") as f:
        chunk_ids: list[str] = pickle.load(f)
    with open(metadata_path, "rb") as f:
        chunk_metadata: dict[str, dict] = pickle.load(f)

    return index, chunk_ids, chunk_metadata


def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunk_ids: list[str],
    top_k: int = 5,
    chunk_metadata: Optional[dict] = None,
    collection=None,
) -> list[FaissSearchResult]:
    vec = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, top_k)
    results: list[FaissSearchResult] = []

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            continue
        cid = chunk_ids[idx]
        metadata: dict = {}

        if collection is not None:
            doc = collection.find_one({"_id": cid})
            if doc:
                metadata = doc

        if not metadata and chunk_metadata is not None:
            metadata = chunk_metadata.get(cid, {})

        results.append(FaissSearchResult(chunk_id=cid, score=float(score), rank=rank, metadata=metadata))

    return results


def query(
    query_text: str,
    index: faiss.Index,
    chunk_ids: list[str],
    embedder,
    top_k: int = 5,
    chunk_metadata: Optional[dict] = None,
    collection=None,
) -> list[FaissSearchResult]:
    query_vec = embedder.embed_texts([query_text])[0]
    return search(query_vec, index, chunk_ids, top_k=top_k, chunk_metadata=chunk_metadata, collection=collection)


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     from ingestion.parser import parse_all_pdfs
#     from ingestion.chunker import chunk_documents
#     from ingestion.embedder import Embedder

#     PDF_DIR = BASE_DIR / "data" / "pdfs"
#     INDEX_DIR = BASE_DIR / "data" / "indexes" / "faiss"

#     print(f"\n{'='*60}")
#     print("FAISS — Phase 2 test")
#     print(f"PDF dir: {PDF_DIR}\n")

#     documents = parse_all_pdfs(PDF_DIR)
#     if not documents:
#         print("No PDFs found.")
#         sys.exit(1)

#     chunks = chunk_documents(documents)
#     embedder = Embedder()
#     embedded = embedder.embed_chunks(chunks)
#     print(f"{len(embedded)} EmbeddedChunks ready.")

#     build_index(embedded, index_dir=INDEX_DIR)
#     index2, cids, meta_map = load_index(INDEX_DIR)
#     print(f"Index: {index2.ntotal} vectors  metadata: {len(meta_map)} entries\n")

#     for q in [
#         "What is the main goal of the National Education Policy 2020?",
#         "What free services are proposed in public hospitals?",
#     ]:
#         print(f"Query: {q}")
#         results = query(q, index2, cids, embedder, top_k=3, chunk_metadata=meta_map)
#         for r in results:
#             print(f"  [{r.rank}] {r.score:.4f}  {r.metadata.get('document_name','?')} p{r.metadata.get('page_number','?')}  {r.metadata.get('text','')[:80]}...")
#         print()

#     print("FAISS test complete.")
