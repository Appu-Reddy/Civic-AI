from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import tiktoken # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore

LOGGER = logging.getLogger(__name__)
TOKENIZER = tiktoken.get_encoding("cl100k_base")
DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "knowledge-base" / "PDFs"


def _count_tokens(text: str) -> int:
	if not text:
		return 0
	return len(TOKENIZER.encode(text))


def _split_sentences(text: str) -> List[str]:
	paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
	sentence_pattern = RecursiveCharacterTextSplitter(
		chunk_size=1,
		chunk_overlap=0,
		length_function=len,
		separators=[". ", "? ", "! ", "; ", ": ", "\n", " ", ""],
	)

	sentences: List[str] = []
	for paragraph in paragraphs:
		if "\n" in paragraph:
			sentences.extend(line.strip() for line in paragraph.splitlines() if line.strip())
			continue

		pieces = sentence_pattern.split_text(paragraph)
		if len(pieces) == 1:
			sentences.append(paragraph)
			continue

		sentences.extend(piece.strip() for piece in pieces if piece.strip())

	return sentences


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=_count_tokens,
		separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
	)
	return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _chunk_code_page(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	token_count = _count_tokens(text)
	if token_count <= max(1, chunk_size * 2):
		return [text.strip()]

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=_count_tokens,
		separators=["\n\n", "\n", " ", ""],
	)
	return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def chunk_pages(pages: Iterable[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> Iterator[Dict]:
	for page in pages:
		text = str(page.get("text", "")).strip()
		if not text:
			continue

		source_file = str(page.get("source_file", "unknown.pdf"))
		page_number = int(page.get("page_number", 0))
		has_code_block = bool(page.get("has_code_block", False))

		if has_code_block:
			chunk_texts = _chunk_code_page(text, chunk_size, chunk_overlap)
		else:
			chunk_texts = _chunk_text(text, chunk_size, chunk_overlap)

		for chunk_index, chunk_text in enumerate(chunk_texts):
			token_count = _count_tokens(chunk_text)
			chunk_id = f"{Path(source_file).stem}_p{page_number}_c{chunk_index}"
			yield {
				"chunk_id": chunk_id,
				"source_file": source_file,
				"page_number": page_number,
				"chunk_index": chunk_index,
				"text": chunk_text,
				"token_count": token_count,
			}


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

	try:
		from parser import parse_directory
	except ImportError:  # pragma: no cover
		from ingestion.parser import parse_directory

	total_chunks = 0
	total_tokens = 0
	for chunk in chunk_pages(parse_directory(str(DEFAULT_PDF_DIR))):
		total_chunks += 1
		total_tokens += int(chunk["token_count"])

	average_tokens = total_tokens / total_chunks if total_chunks else 0.0
	print(f"Chunk count: {total_chunks}")
	print(f"Average token_count: {average_tokens:.2f}")
