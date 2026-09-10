"""
ingest.py — Full ingestion orchestrator: PDF → MongoDB + FAISS + Graph

Usage:
    python -m ingestion.ingest                  # all three stores
    python -m ingestion.ingest --skip-mongo
    python -m ingestion.ingest --skip-faiss
    python -m ingestion.ingest --skip-graph
    python -m ingestion.ingest --rebuild        # clears MongoDB before storing
"""

import argparse
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from ingestion.parser import parse_all_pdfs, ParsedDocument
from ingestion.chunker import chunk_documents, Chunk
from ingestion.embedder import Embedder
from ingestion.processed import (
    save_parsed, save_chunks, load_parsed, load_chunks, is_processed
)
from retrieval.faiss import build_index
from retrieval.graph import build_graph, save_graph
from database.mongodb import get_collection, store_embedded_chunks, clear_collection

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PDF_DIR   = BASE_DIR / "data" / "pdfs"
INDEX_DIR = BASE_DIR / "data" / "indexes" / "faiss"
GRAPH_DIR = BASE_DIR / "data" / "indexes" / "graph"


def run_ingestion(
    pdf_dir: Path = PDF_DIR,
    index_dir: Path = INDEX_DIR,
    graph_dir: Path = GRAPH_DIR,
    processed_dir: Path = BASE_DIR / "data" / "processed",
    skip_mongo: bool = False,
    skip_faiss: bool = False,
    skip_graph: bool = False,
    rebuild: bool = False,
) -> dict:
    summary: dict = {}
    wall_start = time.time()

    # Parse — use cached output unless rebuild
    t0 = time.time()
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}. Aborting.")
        sys.exit(1)

    documents: list[ParsedDocument] = []
    for pdf_file in pdf_files:
        doc_id = pdf_file.stem.lower().replace(" ", "_")
        if not rebuild and is_processed(doc_id, processed_dir):
            cached = load_parsed(doc_id, processed_dir)
            if cached:
                documents.append(cached)
                continue
        from ingestion.parser import parse_pdf
        doc = parse_pdf(pdf_file)
        save_parsed(doc, processed_dir)
        documents.append(doc)

    if not documents:
        logger.error("No documents after parsing. Aborting.")
        sys.exit(1)

    total_pages = sum(d.total_pages for d in documents)
    summary["parse"] = {"documents": len(documents), "pages": total_pages, "elapsed_s": round(time.time() - t0, 2)}
    logger.info("Source Docs Loaded: %d document(s), %d page(s)", len(documents), total_pages)

    # Chunk — use cached output unless rebuild
    t0 = time.time()
    all_chunks: list[Chunk] = []
    for doc in documents:
        if not rebuild and is_processed(doc.document_id, processed_dir):
            cached_chunks = load_chunks(doc.document_id, processed_dir)
            if cached_chunks:
                all_chunks.extend(cached_chunks)
                continue
        from ingestion.chunker import chunk_document
        doc_chunks = chunk_document(doc)
        save_chunks(doc.document_id, doc_chunks, processed_dir)
        all_chunks.extend(doc_chunks)

    chunks = all_chunks
    summary["chunk"] = {"chunks": len(chunks), "elapsed_s": round(time.time() - t0, 2)}
    logger.info("Parsed & Chunked: %d chunk(s)", len(chunks))

    # Embed
    t0 = time.time()
    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)
    summary["embed"] = {"embedded_chunks": len(embedded), "embedding_dim": embedder.embedding_dim, "elapsed_s": round(time.time() - t0, 2)}
    logger.info("Embeddings loaded: %d vector(s)", len(embedded))

    # MongoDB
    if not skip_mongo:
        t0 = time.time()
        try:
            collection = get_collection()
            if rebuild:
                cleared = clear_collection(collection)
            inserted = store_embedded_chunks(embedded, collection)
            total_in_db = collection.count_documents({})
            summary["mongodb"] = {"inserted": inserted, "total_in_collection": total_in_db, "elapsed_s": round(time.time() - t0, 2), "skipped": False}
        except Exception as e:
            logger.error(f"MongoDB storage failed: {e}")
            summary["mongodb"] = {"skipped": True, "error": str(e)}
    else:
        summary["mongodb"] = {"skipped": True}

    # FAISS
    if not skip_faiss:
        t0 = time.time()
        try:
            index_dir.mkdir(parents=True, exist_ok=True)
            faiss_index = build_index(embedded, index_dir=index_dir)
            summary["faiss"] = {"vectors": faiss_index.ntotal, "dim": faiss_index.d, "index_dir": str(index_dir), "elapsed_s": round(time.time() - t0, 2), "skipped": False}
        except Exception as e:
            logger.error(f"FAISS index build failed: {e}")
            summary["faiss"] = {"skipped": True, "error": str(e)}
    else:
        summary["faiss"] = {"skipped": True}

    # Graph
    if not skip_graph:
        t0 = time.time()
        try:
            G = build_graph(chunks)
            save_graph(G, graph_dir=graph_dir)
            summary["graph"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "graph_dir": str(graph_dir), "elapsed_s": round(time.time() - t0, 2), "skipped": False}
        except Exception as e:
            logger.error(f"Graph index build failed: {e}")
            summary["graph"] = {"skipped": True, "error": str(e)}
    else:
        summary["graph"] = {"skipped": True}

    summary["total_elapsed_s"] = round(time.time() - wall_start, 2)
    return summary


def _print_summary(summary: dict) -> None:
    w = 60
    print(f"\n{'='*w}")
    print(f"{'INGESTION COMPLETE':^{w}}")
    print(f"{'='*w}")

    p = summary.get("parse", {})
    c = summary.get("chunk", {})
    e = summary.get("embed", {})
    m = summary.get("mongodb", {})
    f = summary.get("faiss", {})
    g = summary.get("graph", {})

    print(f"\n  Parse   — {p.get('documents','?')} docs, {p.get('pages','?')} pages  ({p.get('elapsed_s','?')}s)  [cached={p.get('cached',0)}]")
    print(f"  Chunk   — {c.get('chunks','?')} chunks  ({c.get('elapsed_s','?')}s)")
    print(f"  Embed   — {e.get('embedded_chunks','?')} chunks  dim={e.get('embedding_dim','?')}  ({e.get('elapsed_s','?')}s)")

    if m.get("skipped"):
        print(f"  MongoDB — SKIPPED{' — ' + m['error'] if 'error' in m else ''}")
    else:
        print(f"  MongoDB — {m.get('inserted','?')} inserted  total={m.get('total_in_collection','?')}  ({m.get('elapsed_s','?')}s)")

    if f.get("skipped"):
        print(f"  FAISS   — SKIPPED{' — ' + f['error'] if 'error' in f else ''}")
    else:
        print(f"  FAISS   — {f.get('vectors','?')} vectors  ({f.get('elapsed_s','?')}s)")

    if g.get("skipped"):
        print(f"  Graph   — SKIPPED{' — ' + g['error'] if 'error' in g else ''}")
    else:
        print(f"  Graph   — {g.get('nodes','?')} nodes  {g.get('edges','?')} edges  ({g.get('elapsed_s','?')}s)")

    print(f"\n  Total   — {summary.get('total_elapsed_s','?')}s")
    print(f"{'='*w}\n")


## TESTING ##
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="DocQA ingestion: PDF → MongoDB + FAISS + Graph")
#     parser.add_argument("--skip-mongo", action="store_true")
#     parser.add_argument("--skip-faiss", action="store_true")
#     parser.add_argument("--skip-graph", action="store_true")
#     parser.add_argument("--rebuild", action="store_true", help="Clear MongoDB before storing")
#     args = parser.parse_args()

#     print(f"\n{'='*60}")
#     print("DOCQA INGESTION PIPELINE")
#     print(f"  PDF dir   : {PDF_DIR}")
#     print(f"  Index dir : {INDEX_DIR}")
#     print(f"  Graph dir : {GRAPH_DIR}")
#     print(f"  MongoDB={('SKIP' if args.skip_mongo else 'ON')}  FAISS={('SKIP' if args.skip_faiss else 'ON')}  Graph={('SKIP' if args.skip_graph else 'ON')}  Rebuild={args.rebuild}")
#     print()

#     summary = run_ingestion(
#         pdf_dir=PDF_DIR, index_dir=INDEX_DIR, graph_dir=GRAPH_DIR,
#         skip_mongo=args.skip_mongo, skip_faiss=args.skip_faiss,
#         skip_graph=args.skip_graph, rebuild=args.rebuild,
#         processed_dir=BASE_DIR / "data" / "processed",
#     )
#     _print_summary(summary)
