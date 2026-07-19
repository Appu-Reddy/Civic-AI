from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore

LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = DEFAULT_BASE_DIR / "knowledge-base" / "PDFs"
OUT_DIR = DEFAULT_BASE_DIR / "knowledge-base" / "Chunks"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 25
BATCH_SIZE = 64
LOG_EVERY_BATCHES = 10

# Swap this default for gte-multilingual in production when the corpus becomes multilingual.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _iter_batches(items: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
	batch: List[Dict] = []
	for item in items:
		batch.append(item)
		if len(batch) >= batch_size:
			yield batch
			batch = []
	if batch:
		yield batch


def embed_chunks(
	chunks: Iterable[Dict],
	model_name: str = MODEL_NAME,
	batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, List[Dict]]:
	model = SentenceTransformer(model_name)
	embedding_batches: List[np.ndarray] = []
	metadata: List[Dict] = []

	total_chunks = len(chunks) if hasattr(chunks, "__len__") else None  # type: ignore[arg-type]
	processed_chunks = 0
	processed_batches = 0

	for batch in _iter_batches(chunks, batch_size):
		texts = [str(chunk.get("text", "")) for chunk in batch]
		batch_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
		if batch_embeddings.dtype != np.float32:
			batch_embeddings = batch_embeddings.astype(np.float32, copy=False)

		embedding_batches.append(batch_embeddings)
		metadata.extend(
			{
				"chunk_id": chunk["chunk_id"],
				"source_file": chunk["source_file"],
				"page_number": int(chunk["page_number"]),
				"chunk_index": int(chunk["chunk_index"]),
				"text": chunk["text"],
				"token_count": int(chunk["token_count"]),
			}
			for chunk in batch
		)

		processed_chunks += len(batch)
		processed_batches += 1
		if processed_batches % LOG_EVERY_BATCHES == 0 or (total_chunks is not None and processed_chunks >= total_chunks):
			if total_chunks is not None:
				LOGGER.info("Embedded %d/%d chunks...", processed_chunks, total_chunks)
			else:
				LOGGER.info("Embedded %d chunks...", processed_chunks)

	if embedding_batches:
		embeddings = np.concatenate(embedding_batches, axis=0)
	else:
		embedding_dim = model.get_sentence_embedding_dimension()
		embeddings = np.empty((0, embedding_dim), dtype=np.float32)

	assert len(embeddings) == len(metadata)
	return embeddings, metadata


def save_embeddings(embeddings: np.ndarray, metadata: List[Dict], out_dir: str) -> None:
	assert len(embeddings) == len(metadata)

	output_path = Path(out_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	np.save(output_path / "embeddings.npy", embeddings.astype(np.float32, copy=False))
	with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
		json.dump(metadata, handle, ensure_ascii=False, indent=2)


def run_pipeline(
	pdf_dir: str,
	out_dir: str,
	chunk_size: int = CHUNK_SIZE,
	chunk_overlap: int = CHUNK_OVERLAP,
	batch_size: int = BATCH_SIZE,
) -> Dict:
	try:
		import parser as parser_module
	except ImportError:  # pragma: no cover
		from ingestion import parser as parser_module

	try:
		from chunker import chunk_pages
	except ImportError:  # pragma: no cover
		from ingestion.chunker import chunk_pages

	pdf_root = Path(pdf_dir)
	total_files = len([path for path in pdf_root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"])
	page_counter = {"count": 0}

	def _tracked_pages() -> Iterator[Dict]:
		for page in parser_module.parse_directory(pdf_dir):
			page_counter["count"] += 1
			yield page

	chunk_stream = chunk_pages(_tracked_pages(), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	embeddings, metadata = embed_chunks(chunk_stream, model_name=MODEL_NAME, batch_size=batch_size)
	save_embeddings(embeddings, metadata, out_dir)

	chunks_per_source = dict(Counter(item["source_file"] for item in metadata))
	summary = {
		"total_files": total_files,
		"total_pages": page_counter["count"],
		"total_chunks": len(metadata),
		"embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
		"chunks_per_source": chunks_per_source,
		"failed_files": sorted(set(parser_module.LAST_FAILED_FILES)),
	}
	return summary


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
	summary = run_pipeline(str(PDF_DIR), str(OUT_DIR), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, batch_size=BATCH_SIZE)
	print(summary)
