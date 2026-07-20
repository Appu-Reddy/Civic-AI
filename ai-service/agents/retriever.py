"""
Retriever agent

Single responsibility: turn a subquery into ranked chunks using Phase-2's
FAISS search. This is NOT an LLM agent - no call_llm() usage anywhere in
this file. It's a tool call: embed -> search -> return.

The embedding model and the FAISS index/metadata are both loaded ONCE at
import time (module level), not per-call - re-loading a sentence-
transformers model or re-reading the index from disk on every retrieval
call is a real cost once queries run per-step across a multi-hop plan.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(_AI_SERVICE_DIR) not in sys.path:
	sys.path.insert(0, str(_AI_SERVICE_DIR))

from sentence_transformers import SentenceTransformer  # type: ignore

from vectorDB.faiss_util import load_index, search  # type: ignore
from ingestion.embedder import MODEL_NAME

from config import DEFAULT_TOP_K, DEFAULT_INDEX_DIR

_EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)

try:
	_INDEX, _ID_MAP = load_index(DEFAULT_INDEX_DIR)
except FileNotFoundError as exc:
	_INDEX, _ID_MAP = None, None
	print(f"[retriever.py] WARNING: FAISS index not loaded at import time: {exc}")


def retrieve(
	subquery: str,
	top_k: int = DEFAULT_TOP_K,
	index_dir: str = DEFAULT_INDEX_DIR,
) -> List[Dict]:
	"""
	Embed `subquery` with the same model/normalization used to build the
	Phase-2 index, then return the top_k ranked chunks.

	Returns the same shape as faiss.search()'s output:
	[{"rank", "score", "chunk_id", "source_file", "page_number", "text"}, ...]
	"""
	if index_dir == DEFAULT_INDEX_DIR:
		if _INDEX is None or _ID_MAP is None:
			raise FileNotFoundError(
				f"No FAISS index loaded from {DEFAULT_INDEX_DIR}. Build it first."
			)
		index, id_map = _INDEX, _ID_MAP
	else:
		index, id_map = load_index(index_dir)

	query_vector = _EMBEDDING_MODEL.encode([subquery], convert_to_numpy=True)[0]
	query_vector = np.asarray(query_vector, dtype=np.float32)

	# search() normalizes the query vector internally (matches how the index
	# was built - see vectorDB/faiss.py's DEFAULT_NORMALIZE), so we pass the
	# raw embedding straight through.
	return search(index, id_map, query_vector, top_k=top_k)


if __name__ == "__main__":
	sample_subquery = "Who is author of python?"
	results = retrieve(sample_subquery, top_k=5)

	print(f"Top {len(results)} results for {sample_subquery!r}:")
	for r in results:
		print(f"  chunk_id={r['chunk_id']!r} score={r['score']:.4f}")