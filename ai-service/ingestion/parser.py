import hashlib
import re
from pathlib import Path

from pypdf import PdfReader  # type: ignore


def clean_text(text):
	return re.sub(r"\s+", " ", text.replace("\x00", "")).strip()


def document_hash(pdf_path):
	return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()


def extract_pages(pdf_path):
	pages = []
	for page_number, page in enumerate(PdfReader(pdf_path).pages, 1):
		text = clean_text(page.extract_text() or "")
		if text:
			pages.append({"page_number": page_number, "text": text})
	return pages


def extract_text(pdf_path):
	return "\n".join(page["text"] for page in extract_pages(pdf_path))