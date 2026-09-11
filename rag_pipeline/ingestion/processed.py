"""
processed.py — Persist and reload parsed/chunked pipeline output.

Saves ParsedDocument pages and Chunk lists as JSON files under
data/processed/<document_id>/ so subsequent runs can skip re-parsing
and re-chunking documents that haven't changed.

File layout:
    data/processed/<document_id>/parsed.json    — pages from parser
    data/processed/<document_id>/chunks.json    — chunks from chunker

Embeddings are NOT stored here — they live in FAISS + MongoDB.
"""

import json
from pathlib import Path

from ingestion.parser import ParsedDocument, ParsedPage
from ingestion.chunker import Chunk

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_parsed(doc: ParsedDocument, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    out_dir = Path(processed_dir) / doc.document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parsed.json"

    data = {
        "document_id": doc.document_id,
        "document_name": doc.document_name,
        "source_path": doc.source_path,
        "total_pages": doc.total_pages,
        "pages": [
            {
                "document_id": p.document_id,
                "document_name": p.document_name,
                "page_number": p.page_number,
                "raw_text": p.raw_text,
                "cleaned_text": p.cleaned_text,
                "section": p.section,
                "word_count": p.word_count,
            }
            for p in doc.pages
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_chunks(
    document_id: str,
    chunks: list[Chunk],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    out_dir = Path(processed_dir) / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "chunks.json"

    data = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page_number": c.page_number,
            "section": c.section,
            "text": c.text,
            "word_count": c.word_count,
            "char_offset": c.char_offset,
            "domain": c.domain,
            "source": c.source,
        }
        for c in chunks
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_parsed(document_id: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> ParsedDocument | None:
    path = Path(processed_dir) / document_id / "parsed.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    pages = [
        ParsedPage(
            document_id=p["document_id"],
            document_name=p["document_name"],
            page_number=p["page_number"],
            raw_text=p["raw_text"],
            cleaned_text=p["cleaned_text"],
            section=p.get("section"),
            word_count=p["word_count"],
        )
        for p in data["pages"]
    ]
    return ParsedDocument(
        document_id=data["document_id"],
        document_name=data["document_name"],
        source_path=data["source_path"],
        total_pages=data["total_pages"],
        pages=pages,
    )


def load_chunks(document_id: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> list[Chunk] | None:
    path = Path(processed_dir) / document_id / "chunks.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            document_name=c["document_name"],
            page_number=c["page_number"],
            section=c.get("section"),
            text=c["text"],
            word_count=c["word_count"],
            char_offset=c["char_offset"],
            domain=c["domain"],
            source=c["source"],
        )
        for c in data
    ]


def is_processed(document_id: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> bool:
    """Return True if both parsed.json and chunks.json exist for this document."""
    base = Path(processed_dir) / document_id
    return (base / "parsed.json").exists() and (base / "chunks.json").exists()


def list_processed(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> list[str]:
    """Return list of document_ids that have saved processed output."""
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        return []
    return [d.name for d in sorted(processed_dir.iterdir()) if d.is_dir()]