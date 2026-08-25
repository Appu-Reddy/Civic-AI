import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
import re

import tiktoken # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore

TOKENIZER = tiktoken.get_encoding("cl100k_base")

ARTICLE_HEADING_PATTERN = re.compile(
    r"^\s*(\d+[A-Z]?)\.\s+[A-Z][a-zA-Z\s]+[—\-]",
    re.MULTILINE,
)

def _count_tokens(text: str) -> int:
	if not text:
		return 0
	return len(TOKENIZER.encode(text))


def split_by_article(text: str) -> List[Dict[str, Optional[str]]]:
    matches = list(ARTICLE_HEADING_PATTERN.finditer(text))

    if not matches:
        return [{"article_ref": None, "text": text}]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
                "article_ref": match.group(1),
                "text": text[start:end].strip(),
            }
        )

    return sections


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


def chunk_pages(pages: Iterable[Dict], chunk_size: int = 250, chunk_overlap: int = 40) -> Iterator[Dict]:
	for page in pages:
		text = str(page.get("text", "")).strip()
		if not text:
			continue

		source_file = str(page.get("source_file", "unknown.pdf"))
		page_number = int(page.get("page_number", 0))
		has_code_block = bool(page.get("has_code_block", False))

		article_sections = split_by_article(text)

		for section in article_sections:
			article_ref = section["article_ref"]
			section_text = section["text"]

			if not section_text:
				continue

			token_count = _count_tokens(section_text)

			# Only recursively split oversized sections
			if token_count <= chunk_size:
				chunk_texts = [section_text.strip()]
			else:
				if has_code_block:
					chunk_texts = _chunk_code_page(
						section_text,
						chunk_size,
						chunk_overlap,
					)
				else:
					chunk_texts = _chunk_text(
						section_text,
						chunk_size,
						chunk_overlap,
					)

			article_suffix = (
				f"_a{article_ref}" if article_ref is not None else ""
			)

			for chunk_index, chunk_text in enumerate(chunk_texts):
				yield {
					"chunk_id": (
						f"{Path(source_file).stem}"
						f"_p{page_number}"
						f"{article_suffix}"
						f"_c{chunk_index}"
					),
					"source_file": source_file,
					"page_number": page_number,
					"article_ref": article_ref,
					"chunk_index": chunk_index,
					"text": chunk_text,
					"token_count": _count_tokens(chunk_text),
				}


# TESTING:

# DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "knowledge-base" / "PDFs"
# if __name__ == "__main__":
# 	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# 	try:
# 		from parser import parse_directory
# 	except ImportError:  # pragma: no cover
# 		from ingestion.parser import parse_directory

# 	total_chunks = 0
# 	total_tokens = 0
# 	for chunk in chunk_pages(parse_directory(str(DEFAULT_PDF_DIR))):
# 		total_chunks += 1
# 		total_tokens += int(chunk["token_count"])

# 	average_tokens = total_tokens / total_chunks if total_chunks else 0.0
# 	print(f"Chunk count: {total_chunks}")
# 	print(f"Average token_count: {average_tokens:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_pages = [
        # 1. Normal text
        {
            "source_file": "normal.pdf",
            "page_number": 1,
            "has_code_block": False,
            "text": (
                "Machine learning is a branch of artificial intelligence. "
                "It enables computers to learn from data without being explicitly programmed. "
                * 25
            ),
        },

        # 2. Legal document with articles
        {
            "source_file": "constitution.pdf",
            "page_number": 5,
            "has_code_block": False,
            "text": """
				14. Equality before law.—
				The State shall not deny to any person equality before the law or the equal protection
				of the laws within the territory of India.

				15. Prohibition of discrimination.—
				The State shall not discriminate against any citizen on grounds only of religion,
				race, caste, sex or place of birth.

				16. Equality of opportunity in matters of public employment.—
				There shall be equality of opportunity for all citizens in matters relating to
				employment or appointment to any office under the State.
				""" * 5,
        },

        # 3. Code page
        {
            "source_file": "python_guide.pdf",
            "page_number": 10,
            "has_code_block": True,
            "text": """
			def fibonacci(n):
				if n <= 1:
					return n
				return fibonacci(n - 1) + fibonacci(n - 2)

			for i in range(10):
				print(fibonacci(i))

			class Person:
				def __init__(self, name):
					self.name = name

				def greet(self):
					print(f"Hello {self.name}")

			""" * 20,
        },
    ]

    print("=" * 80)
    print("TESTING CHUNKER")
    print("=" * 80)

    total_chunks = 0

    for chunk in chunk_pages(test_pages, chunk_size=100, chunk_overlap=20):
        total_chunks += 1

        print(f"\nChunk #{total_chunks}")
        print("-" * 60)
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print(f"Source File : {chunk['source_file']}")
        print(f"Page        : {chunk['page_number']}")
        print(f"Article Ref : {chunk['article_ref']}")
        print(f"Chunk Index : {chunk['chunk_index']}")
        print(f"Tokens      : {chunk['token_count']}")
        print("Text Preview:")
        print(chunk["text"][:250].replace("\n", " "))
        print("-" * 60)

    print("\n" + "=" * 80)
    print(f"Total chunks generated: {total_chunks}")
