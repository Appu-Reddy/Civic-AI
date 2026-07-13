from pathlib import Path


def chunk_text(text, chunk_size=200, overlap=50):
	words = text.split()
	step = chunk_size - overlap
	return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step) if words[i:i + chunk_size]]


def chunk_document(pages, chunk_size=200, overlap=50, source_pdf="", document_hash=""):
	words = [(word, page["page_number"]) for page in pages for word in page["text"].split()]
	prefix = document_hash[:12] if document_hash else (Path(source_pdf).stem if source_pdf else "document")
	step = chunk_size - overlap
	chunks = []
	for chunk_index, i in enumerate(range(0, len(words), step)):
		chunk_words = words[i:i + chunk_size]
		if not chunk_words:
			break
		chunks.append({"chunk_id": f"{prefix}:p{chunk_words[0][1]}-{chunk_words[-1][1]}:c{chunk_index}", "chunk_index": chunk_index, "start_page": chunk_words[0][1], "end_page": chunk_words[-1][1], "source_pdf": source_pdf, "document_hash": document_hash, "text": " ".join(word for word, _ in chunk_words)})
		if i + chunk_size >= len(words):
			break
	return chunks


chunk_pages = chunk_document
