import json
from pathlib import Path

import faiss
import numpy as np

from ingestion.chunker import chunk_document
from ingestion.embedder import MODEL_NAME, generate_embeddings, load_model
from ingestion.parser import document_hash, extract_pages


def load_metadata(path):
	path = Path(path)
	if not path.exists():
		return {"schema_version": 1, "documents": {}, "chunks": []}
	return json.loads(path.read_text(encoding="utf-8"))


def save_metadata(metadata, path):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def build_index(embeddings):
	vectors = np.asarray(embeddings, dtype=np.float32).copy()
	if len(vectors)==0: 
		raise ValueError("Vectors not found")
	faiss.normalize_L2(vectors)
	index = faiss.IndexFlatIP(vectors.shape[1])
	index.add(vectors)
	return index


def add_embeddings(index, embeddings):
	vectors = np.asarray(embeddings, dtype=np.float32).copy()
	faiss.normalize_L2(vectors)
	index.add(vectors)
	return index


def save_index(index, path):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(path))


def load_index(path):
	if not Path(path).exists(): 
		raise FileNotFoundError("File Not Found")
	return faiss.read_index(str(path))


def search(index, query_embedding, chunks, top_k=5):
	query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
	faiss.normalize_L2(query)
	scores, indices = index.search(query, min(top_k, index.ntotal))
	results = []
	for score, i in zip(scores[0], indices[0]):
		chunk = dict(chunks[i])
		if "start_page" not in chunk or "end_page" not in chunk:
			chunk["start_page"] = chunk.get("page_number")
			chunk["end_page"] = chunk.get("page_number")
		chunk["score"] = float(score)
		chunk["vector_index"] = int(i)
		results.append(chunk)
	return results


def ingest_document(pdf_path, index_path, metadata_path=None, chunk_size=200, overlap=50, model_name=MODEL_NAME, model=None):
	pdf_path = Path(pdf_path)
	index_path = Path(index_path)
	metadata_path = Path(metadata_path) if metadata_path else index_path.with_name("metadata.json")
	doc_hash = document_hash(pdf_path)
	metadata = load_metadata(metadata_path)
	if doc_hash in metadata["documents"]:
		return {"status": "duplicate", "document_hash": doc_hash, "source_pdf": str(pdf_path.resolve()), "index_path": str(index_path), "metadata_path": str(metadata_path), "chunk_count": 0, "vector_count": len(metadata["chunks"])}

	pages = extract_pages(pdf_path)
	chunks = chunk_document(pages, chunk_size, overlap, str(pdf_path.resolve()), doc_hash)
	if model is None:
		model = load_model(model_name)

	if chunks:
		embeddings = generate_embeddings([chunk["text"] for chunk in chunks], model)
		index = load_index(index_path) if index_path.exists() else build_index(embeddings)
		if index_path.exists():
			index = add_embeddings(index, embeddings)
		start = index.ntotal - len(chunks)
		for offset, chunk in enumerate(chunks):
			chunk["vector_index"] = start + offset
	else:
		index = load_index(index_path) if index_path.exists() else None

	metadata["documents"][doc_hash] = {"document_hash": doc_hash, "source_pdf": str(pdf_path.resolve()), "page_count": len(pages), "chunk_count": len(chunks), "chunk_size": chunk_size, "overlap": overlap}
	metadata["chunks"].extend(chunks)
	if index is not None:
		save_index(index, index_path)
	save_metadata(metadata, metadata_path)
	return {"status": "ingested", "document_hash": doc_hash, "source_pdf": str(pdf_path.resolve()), "index_path": str(index_path), "metadata_path": str(metadata_path), "chunk_count": len(chunks), "vector_count": index.ntotal if index is not None else 0}


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices={"build", "search", "ingest"}, required=True)
	parser.add_argument("--index-path", default="index.faiss")
	parser.add_argument("--metadata-path")
	parser.add_argument("--embeddings-path")
	parser.add_argument("--query-path")
	parser.add_argument("--pdf-path")
	parser.add_argument("--chunk-size", type=int, default=200)
	parser.add_argument("--overlap", type=int, default=50)
	parser.add_argument("--model-name", default=MODEL_NAME)
	parser.add_argument("--top-k", type=int, default=5)
	args = parser.parse_args()

	if args.mode == "build":
		save_index(build_index(np.load(args.embeddings_path)), args.index_path)
	elif args.mode == "ingest":
		print(json.dumps(ingest_document(args.pdf_path, args.index_path, args.metadata_path, args.chunk_size, args.overlap, args.model_name), indent=2, ensure_ascii=False))
	else:
		metadata = load_metadata(args.metadata_path or Path(args.index_path).with_name("metadata.json"))
		print(json.dumps(search(load_index(args.index_path), np.load(args.query_path), metadata["chunks"], args.top_k), indent=2, ensure_ascii=False))
