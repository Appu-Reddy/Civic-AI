import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)

_HEADING_PATTERNS = [
    re.compile(r"^\s{0,4}(?:CHAPTER|SECTION|PART)\s+[IVXLC\d]+", re.IGNORECASE),
    re.compile(r"^\s{0,4}\d+\.\s+[A-Z][A-Za-z\s]{3,60}$"),
    re.compile(r"^\s{0,4}[A-Z][A-Z\s]{4,60}$"),
    re.compile(r"^\s{0,4}[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,6}\s*$"),
]


@dataclass
class ParsedPage:
    document_id: str
    document_name: str
    page_number: int
    raw_text: str
    cleaned_text: str
    section: Optional[str]
    word_count: int


@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    source_path: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)


def _detect_section(page_text: str) -> Optional[str]:
    lines = [l.strip() for l in page_text.splitlines() if l.strip()][:20]
    for line in lines:
        for pattern in _HEADING_PATTERNS:
            if pattern.match(line) and len(line) <= 120:
                return line
    return None


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\x00", "").replace("\x0c", "\n")
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_name = pdf_path.stem
    document_id = document_name.lower().replace(" ", "_")
    parsed_pages: list[ParsedPage] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = _clean_text(raw_text)
            parsed_pages.append(ParsedPage(
                document_id=document_id,
                document_name=document_name,
                page_number=i,
                raw_text=raw_text,
                cleaned_text=cleaned,
                section=_detect_section(cleaned),
                word_count=len(cleaned.split()),
            ))

    return ParsedDocument(
        document_id=document_id,
        document_name=document_name,
        source_path=str(pdf_path),
        total_pages=total_pages,
        pages=parsed_pages,
    )


def parse_all_pdfs(pdf_dir: str | Path) -> list[ParsedDocument]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    documents = []
    for pdf_file in pdf_files:
        try:
            documents.append(parse_pdf(pdf_file))
        except Exception as e:
            logger.error(f"Failed to parse {pdf_file.name}: {e}")
    return documents


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     PDF_DIR = BASE_DIR / "data" / "pdfs"

#     print(f"\n{'='*60}")
#     print("PARSER — Phase 1 test")
#     print(f"PDF dir: {PDF_DIR}\n")

#     documents = parse_all_pdfs(PDF_DIR)

#     for doc in documents:
#         non_empty = [p for p in doc.pages if p.word_count > 0]
#         total_words = sum(p.word_count for p in doc.pages)
#         sections_found = [p.section for p in doc.pages if p.section]
#         print(f"Document : {doc.document_name}  ({doc.total_pages} pages)")
#         print(f"  Non-empty pages  : {len(non_empty)}")
#         print(f"  Total words      : {total_words:,}")
#         print(f"  Sections detected: {len(sections_found)}")
#         for page in doc.pages:
#             if page.word_count > 10:
#                 print(f"  Sample (p{page.page_number}): {page.cleaned_text[:200]}...")
#                 break

#     if not documents:
#         print("No documents parsed.")
#         sys.exit(1)

#     print(f"\nDone. {len(documents)} document(s) parsed.")
