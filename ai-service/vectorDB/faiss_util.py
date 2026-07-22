import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss  # type: ignore

LOGGER = logging.getLogger(__name__)


# Config

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_DIR = DEFAULT_BASE_DIR / "knowledge-base" / "Chunks"
FAISS_DIR = DEFAULT_BASE_DIR / "knowledge-base" / "FAISS"

INDEX_FILENAME = "index.faiss"
ID_MAP_FILENAME = "id_map.json"

# A flat (exhaustive) index gives EXACT search results and is simple to
# reason about. It's fast enough up to roughly tens of thousands of vectors
# — comfortably covers "dozens of ~300-page PDFs" (likely low tens of
# thousands of chunks). If the corpus grows well past that and search
# latency becomes a problem, swap IndexFlatIP/IndexFlatL2 below for
# faiss.IndexIVFFlat (needs a .train() step) or faiss.IndexHNSWFlat —
# build_faiss_index() is the only function that needs to change.
DEFAULT_NORMALIZE = True  # True = cosine similarity via inner product


# Load Embeddings

def load_embeddings_and_metadata(chunks_dir: str) -> Tuple[np.ndarray, List[Dict]]:
	chunks_path = Path(chunks_dir)
	embeddings_path = chunks_path / "embeddings.npy"
	metadata_path = chunks_path / "metadata.json"

	if not embeddings_path.exists() or not metadata_path.exists():
		raise FileNotFoundError(
			f"Expected embeddings.npy and metadata.json in {chunks_dir}. Run embedder.py first."
		)

	embeddings = np.load(embeddings_path)
	with metadata_path.open("r", encoding="utf-8") as handle:
		metadata = json.load(handle)

	if len(embeddings) != len(metadata):
		raise ValueError(
			f"Alignment mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata records. "
			"embeddings.npy and metadata.json must be row-aligned (see embedder.py's own assertion)."
		)

	return embeddings.astype(np.float32, copy=False), metadata


def _normalize(vectors: np.ndarray) -> np.ndarray:
	"""L2-normalize rows so inner product == cosine similarity."""
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	norms[norms == 0] = 1.0  # guard against degenerate zero vectors
	return (vectors / norms).astype(np.float32, copy=False)


def build_faiss_index(embeddings: np.ndarray, normalize: bool = DEFAULT_NORMALIZE) -> faiss.Index:
	"""
	Build a flat, exact FAISS index over the given embeddings.

	normalize=True (default) makes the index compute cosine similarity via
	inner product — the metric most sentence-embedding models (including
	all-MiniLM-L6-v2) are tuned against. The vectors on disk in
	embeddings.npy are left untouched; normalization happens on a copy here.
	"""
	if embeddings.ndim != 2 or embeddings.shape[0] == 0:
		raise ValueError("embeddings must be a non-empty 2D array of shape (num_chunks, embedding_dim)")

	dim = embeddings.shape[1]
	vectors = _normalize(embeddings) if normalize else embeddings.astype(np.float32, copy=False)

	index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
	index.add(vectors)
	return index


def save_index(index: faiss.Index, metadata: List[Dict], out_dir: str) -> None:
	"""
	Persist the FAISS index plus a row_id -> chunk lookup ("id_map.json").

	metadata.json (owned by embedder.py) already holds the full text/token
	counts and is row-aligned with embeddings.npy. id_map.json restates the
	same row -> chunk_id/source_file/page_number/text mapping alongside the
	index file, so a raw FAISS search hit (just a row id + score) can be
	resolved straight to something usable without re-opening Chunks/.
	"""
	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	faiss.write_index(index, str(out_path / INDEX_FILENAME))

	id_map = [
		{
			"row_id": i,
			"chunk_id": item["chunk_id"],
			"source_file": item["source_file"],
			"page_number": item["page_number"],
			"text": item["text"],
		}
		for i, item in enumerate(metadata)
	]
	with (out_path / ID_MAP_FILENAME).open("w", encoding="utf-8") as handle:
		json.dump(id_map, handle, ensure_ascii=False, indent=2)


def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
	"""Load a previously saved index + its id_map."""
	index_path = Path(index_dir)
	index_file = index_path / INDEX_FILENAME
	id_map_file = index_path / ID_MAP_FILENAME

	if not index_file.exists() or not id_map_file.exists():
		raise FileNotFoundError(f"No FAISS index found in {index_dir}. Run build_index() first.")

	index = faiss.read_index(str(index_file))
	with id_map_file.open("r", encoding="utf-8") as handle:
		id_map = json.load(handle)

	return index, id_map


def build_index(
	chunks_dir: str = str(CHUNKS_DIR),
	out_dir: str = str(FAISS_DIR),
	normalize: bool = DEFAULT_NORMALIZE,
) -> Dict:
	"""
	End-to-end: load embeddings/metadata from Chunks/, build the index, save
	it to FAISS/. Returns a small summary dict for logging/inspection.
	"""
	embeddings, metadata = load_embeddings_and_metadata(chunks_dir)
	index = build_faiss_index(embeddings, normalize=normalize)
	save_index(index, metadata, out_dir)

	summary = {
		"total_vectors": index.ntotal,
		"embedding_dim": embeddings.shape[1],
		"index_type": "IndexFlatIP (cosine)" if normalize else "IndexFlatL2",
		"out_dir": str(out_dir),
	}
	LOGGER.info("Built FAISS index: %s", summary)
	return summary


# Query-time helpers

def embed_query(text: str, model_name: Optional[str] = None) -> np.ndarray:
	"""
	Embed a single query string with the SAME model embedder.py used, so the
	query vector lives in the same space as the indexed chunk vectors.
	Imported lazily so importing/loading/searching an existing index doesn't
	require pulling in sentence-transformers or downloading model weights.
	"""
	from sentence_transformers import SentenceTransformer  # type: ignore

	if model_name is None:
		try:
			from embedder import MODEL_NAME as model_name  # type: ignore
		except ImportError:  # pragma: no cover
			from ingestion.embedder import MODEL_NAME as model_name  # type: ignore

	model = SentenceTransformer(model_name)
	vector = model.encode([text], convert_to_numpy=True)[0]
	return vector.astype(np.float32, copy=False)


def search(
	index: faiss.Index,
	id_map: List[Dict],
	query_embedding: np.ndarray,
	top_k: int = 5,
	normalize: bool = DEFAULT_NORMALIZE,
) -> List[Dict]:
	"""
	Search the index for the top_k nearest chunks to query_embedding.
	Returns results ordered best-first:
	[{"rank", "score", "chunk_id", "source_file", "page_number", "text"}, ...]
	"""
	if index.ntotal == 0:
		return []

	query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
	if normalize:
		query = _normalize(query)

	scores, row_ids = index.search(query, min(top_k, index.ntotal))

	results: List[Dict] = []
	for rank, (row_id, score) in enumerate(zip(row_ids[0], scores[0])):
		if row_id == -1:
			continue
		entry = id_map[row_id]
		results.append(
			{
				"rank": rank,
				"score": float(score),
				"chunk_id": entry["chunk_id"],
				"source_file": entry["source_file"],
				"page_number": entry["page_number"],
				"text": entry["text"],
			}
		)
	return results


# TESTING:

# if __name__ == "__main__":
# 	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# 	summary = build_index(str(CHUNKS_DIR), str(FAISS_DIR))
# 	print(summary)

# 	if summary["total_vectors"] > 0:
# 		index, id_map = load_index(str(FAISS_DIR))
# 		sample_query = "How to run a code in python?"
# 		query_vector = embed_query(sample_query)
# 		results = search(index, id_map, query_vector, top_k=3)
# 		print(f"Sample search results for {sample_query!r}:")
# 		for r in results:
# 			print(r)