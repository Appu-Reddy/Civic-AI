import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List

import fitz # type: ignore

LOGGER = logging.getLogger(__name__)

PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?\d+(?:\s*/\s*\d+|\s*of\s*\d+)?$", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

LAST_FAILED_FILES: List[str] = []
LAST_TOTAL_FILES: int = 0
LAST_TOTAL_PAGES: int = 0


def _normalize_text(text: str) -> str:
	return WHITESPACE_PATTERN.sub(" ", text).strip().lower()


def _looks_like_page_number(text: str) -> bool:
	return bool(PAGE_NUMBER_PATTERN.match(text.strip()))


def _is_code_like(lines: List[str]) -> bool:
	if not lines:
		return False

	code_signals = 0
	for line in lines:
		stripped = line.lstrip()
		if not stripped:
			continue
		if line.startswith(("    ", "\t")):
			code_signals += 1
		if any(token in stripped for token in ("{", "}", ";", "#include", "public ", "private ", "class ", "def ", "import ", "using namespace", "std::", "->")):
			code_signals += 1

	return code_signals >= max(1, len(lines) // 3)


def _postprocess_text(text: str, has_code_block: bool) -> str:
    if not text:
        return ""

    if has_code_block:
        return text.strip()

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _reflow_paragraph(lines: List[str]) -> str:
	if not lines:
		return ""

	if _is_code_like(lines):
		return "\n".join(line.rstrip() for line in lines).strip()

	paragraph = ""
	for line in lines:
		stripped = line.strip()
		if not stripped:
			continue
		if not paragraph:
			paragraph = stripped
		elif paragraph.endswith("-"):
			paragraph = paragraph[:-1] + stripped
		else:
			paragraph += " " + stripped

	return paragraph.strip()


def _clean_page_lines(lines: List[str], repeated_headers: set[str], repeated_footers: set[str]) -> List[str]:
	cleaned_lines: List[str] = []
	paragraph_buffer: List[str] = []

	for raw_line in lines:
		line = raw_line.rstrip()
		stripped = line.strip()

		if not stripped:
			if paragraph_buffer:
				paragraph = _reflow_paragraph(paragraph_buffer)
				if paragraph:
					cleaned_lines.append(paragraph)
				paragraph_buffer = []
			continue

		normalized = _normalize_text(stripped)
		if not normalized:
			continue
		if _looks_like_page_number(normalized):
			continue
		if normalized in repeated_headers or normalized in repeated_footers:
			continue

		paragraph_buffer.append(line)

	if paragraph_buffer:
		paragraph = _reflow_paragraph(paragraph_buffer)
		if paragraph:
			cleaned_lines.append(paragraph)

	return cleaned_lines


def _record_failed_file(pdf_path: str) -> None:
	if pdf_path not in LAST_FAILED_FILES:
		LAST_FAILED_FILES.append(pdf_path)


def parse_pdf(pdf_path: str) -> List[Dict]:
	pages: List[Dict] = []

	try:
		document = fitz.open(pdf_path)
	except Exception as exc:  # noqa: BLE001
		LOGGER.exception("Failed to open PDF %s: %s", pdf_path, exc)
		_record_failed_file(pdf_path)
		return pages

	page_lines: List[tuple[int, List[str]]] = []
	header_counter: Counter[str] = Counter()
	footer_counter: Counter[str] = Counter()
	encountered_error = False

	try:
		for page_number, page in enumerate(document, start=1):
			try:
				raw_text = page.get_text("text", sort=True) or ""
				lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
				non_empty = [line.strip() for line in lines if line.strip()]

				if non_empty:
					for candidate in non_empty[:2]:
						header_counter[_normalize_text(candidate)] += 1
					for candidate in non_empty[-2:]:
						footer_counter[_normalize_text(candidate)] += 1

				page_lines.append((page_number, lines))
			except Exception as exc:  # noqa: BLE001
				encountered_error = True
				LOGGER.exception("Failed to extract page %s from %s: %s", page_number, pdf_path, exc)

		threshold = max(2, math.ceil(len(page_lines) * 0.5))
		repeated_headers = {
			text for text, count in header_counter.items() if count >= threshold and text and not _looks_like_page_number(text)
		}
		repeated_footers = {
			text for text, count in footer_counter.items() if count >= threshold and text and not _looks_like_page_number(text)
		}

		for page_number, lines in page_lines:
			try:
				cleaned_lines = _clean_page_lines(lines, repeated_headers, repeated_footers)
				text = "\n\n".join(cleaned_lines).strip()
				has_code_block = any(_is_code_like(chunk.split("\n")) for chunk in cleaned_lines)
				text = _postprocess_text(text, has_code_block)

				if text:
					pages.append(
						{
							"page_number": page_number,
							"source_file": os.path.basename(pdf_path),
							"text": text,
							"has_code_block": has_code_block,
						}
					)
			except Exception as exc:  # noqa: BLE001
				encountered_error = True
				LOGGER.exception("Failed to clean page %s from %s: %s", page_number, pdf_path, exc)

	finally:
		document.close()

	if encountered_error:
		_record_failed_file(pdf_path)

	return pages


def parse_directory(dir_path: str) -> Iterator[Dict]:
	global LAST_FAILED_FILES, LAST_TOTAL_FILES, LAST_TOTAL_PAGES

	LAST_FAILED_FILES.clear()
	LAST_TOTAL_FILES = 0
	LAST_TOTAL_PAGES = 0

	pdf_root = Path(dir_path)
	pdf_files = sorted(path for path in pdf_root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
	LAST_TOTAL_FILES = len(pdf_files)

	for pdf_path in pdf_files:
		pages = parse_pdf(str(pdf_path))
		for page in pages:
			LAST_TOTAL_PAGES += 1
			yield page
