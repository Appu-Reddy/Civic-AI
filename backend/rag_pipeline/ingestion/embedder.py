import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.parser import parse_all_pdfs
from ingestion.chunker import Chunk, chunk_documents

DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: int = 64


@dataclass
class EmbeddedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    section: Optional[str]
    text: str
    word_count: int
    char_offset: int
    domain: str
    source: str
    embedding: np.ndarray
    embedding_model: str

    @classmethod
    def from_chunk(cls, chunk: Chunk, embedding: np.ndarray, model_name: str) -> "EmbeddedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            section=chunk.section,
            text=chunk.text,
            word_count=chunk.word_count,
            char_offset=chunk.char_offset,
            domain=chunk.domain,
            source=chunk.source,
            embedding=embedding,
            embedding_model=model_name,
        )


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        start = time.time()
        self._model = SentenceTransformer(model_name)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = DEFAULT_BATCH_SIZE) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        return [EmbeddedChunk.from_chunk(chunk, embeddings[i], self.model_name) for i, chunk in enumerate(chunks)]


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    return Embedder(model_name=model_name).embed_chunks(chunks, batch_size=batch_size)


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     from ingestion.parser import parse_all_pdfs
#     from ingestion.chunker import chunk_documents
#     PDF_DIR = BASE_DIR / "data" / "pdfs"

#     print(f"\n{'='*60}")
#     print("EMBEDDER — Phase 1 test")
#     print(f"PDF dir: {PDF_DIR}  model: {DEFAULT_MODEL_NAME}\n")

#     documents = parse_all_pdfs(PDF_DIR)
#     if not documents:
#         print("No documents found.")
#         sys.exit(1)

#     chunks = chunk_documents(documents)
#     embedder = Embedder()
#     embedded = embedder.embed_chunks(chunks)

#     print(f"EmbeddedChunks : {len(embedded)}")
#     print(f"Dim            : {embedder.embedding_dim}")
#     norms = [float(np.linalg.norm(e.embedding)) for e in embedded[:5]]
#     print(f"L2 norms (x5)  : {[round(n, 4) for n in norms]}")
#     if len(embedded) >= 2:
#         print(f"Cosine sim 0v1 : {float(np.dot(embedded[0].embedding, embedded[1].embedding)):.4f}")

#     ec = embedded[0]
#     print(f"\nSample: {ec.document_name} p{ec.page_number} | {ec.domain}")
#     print(f"  {ec.text[:150]}...")

#     print(f"\nDone. {len(embedded)} EmbeddedChunks ready.")
