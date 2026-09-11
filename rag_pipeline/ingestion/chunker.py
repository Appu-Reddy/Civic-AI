import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ingestion.parser import ParsedDocument, parse_all_pdfs

DEFAULT_CHUNK_SIZE_WORDS: int = 150
DEFAULT_OVERLAP_WORDS: int = 30
MIN_PAGE_WORDS: int = 20

_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


@dataclass
class Chunk:
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


def _infer_domain(document_name: str) -> str:
    """
    Derive a domain label directly from the document filename stem.
    The document name (without extension) is lowercased and used as the domain
    so that any PDF — regardless of subject — gets a stable, meaningful label
    without requiring a hard-coded keyword list.

    Examples:
        "Climate_Report_2023"  → "climate_report_2023"
        "climate_report_2023" → "climate_report_2023"
        ""           → "general"
    """
    slug = document_name.strip().lower().replace(" ", "_")
    return slug if slug else "general"


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_END.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    sentence_word_counts = [len(s.split()) for s in sentences]
    chunks: list[tuple[str, int]] = []
    current_sentences: list[str] = []
    current_word_count: int = 0
    consumed_chars: int = 0

    for sent_idx, sentence in enumerate(sentences):
        current_sentences.append(sentence)
        current_word_count += sentence_word_counts[sent_idx]

        if current_word_count >= chunk_size:
            chunks.append((" ".join(current_sentences), consumed_chars))

            overlap_sentences: list[str] = []
            overlap_words: int = 0
            for s in reversed(current_sentences):
                w = len(s.split())
                if overlap_words + w <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_words += w
                else:
                    break

            dropped = current_sentences[: len(current_sentences) - len(overlap_sentences)]
            consumed_chars += sum(len(s) + 1 for s in dropped)
            current_sentences = overlap_sentences
            current_word_count = overlap_words

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if len(chunk_text.split()) >= 10:
            chunks.append((chunk_text, consumed_chars))

    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    source_filename = Path(doc.source_path).name
    all_chunks: list[Chunk] = []

    for page in doc.pages:
        if page.word_count < MIN_PAGE_WORDS:
            continue
        for chunk_text, char_offset in _chunk_text(page.cleaned_text, chunk_size, overlap):
            all_chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc.document_id,
                document_name=doc.document_name,
                page_number=page.page_number,
                section=page.section,
                text=chunk_text,
                word_count=len(chunk_text.split()),
                char_offset=char_offset,
                domain=_infer_domain(doc.document_name),
                source=source_filename,
            ))

    return all_chunks


def chunk_documents(
    documents: list[ParsedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks


## TESTING ##
# if __name__ == "__main__":
#     import sys
#     from collections import Counter

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     from ingestion.parser import parse_all_pdfs
#     PDF_DIR = BASE_DIR / "data" / "pdfs"

#     print(f"\n{'='*60}")
#     print("CHUNKER — Phase 1 test")
#     print(f"PDF dir: {PDF_DIR}\n")

#     documents = parse_all_pdfs(PDF_DIR)
#     if not documents:
#         print("No documents found.")
#         sys.exit(1)

#     chunks = chunk_documents(documents)

#     doc_counts = Counter(c.document_id for c in chunks)
#     domain_counts = Counter(c.domain for c in chunks)
#     word_counts = [c.word_count for c in chunks]

#     for doc_id, count in doc_counts.items():
#         print(f"  {doc_id:<40} {count:>5} chunks")
#     print(f"\nDomains: {dict(domain_counts)}")
#     print(f"Word count — min:{min(word_counts)}  max:{max(word_counts)}  avg:{sum(word_counts)/len(word_counts):.1f}")

#     print(f"\nSample (first chunk):")
#     c = chunks[0]
#     print(f"  {c.document_name} p{c.page_number} | {c.domain} | {c.word_count} words")
#     print(f"  {c.text[:200]}...")

#     print(f"\nDone. {len(chunks)} chunks produced.")
