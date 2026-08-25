import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore

LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = DEFAULT_BASE_DIR / "knowledge-base" / "PDFs"
OUT_DIR = DEFAULT_BASE_DIR / "knowledge-base" / "Chunks"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40
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
		embedding_dim = model.get_embedding_dimension()
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