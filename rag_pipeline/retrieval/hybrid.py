import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RRF_K: int = 60
FAISS_TOP_K: int = 10
GRAPH_TOP_K: int = 10
FINAL_TOP_K: int = 5

_STOPWORDS = {
    # question words and articles
    "what", "which", "when", "where", "who", "how", "why",
    "the", "a", "an",
    # common verbs / auxiliaries
    "are", "can", "for", "and", "that", "this", "with", "from", "into",
    "have", "does", "will", "been", "some", "also", "more", "such",
    "was", "were", "has", "had", "not", "but", "its", "their",
    # common prepositions / conjunctions
    "about", "over", "under", "after", "before", "between", "through",
    "there", "here", "they", "them", "these", "those",
}


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    section: Optional[str]
    domain: str
    source: str
    rrf_score: float
    rank: int
    faiss_rank: Optional[int]
    graph_rank: Optional[int]
    retrieval_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "text": self.text,
            "document_name": self.document_name, "page_number": self.page_number,
            "section": self.section, "domain": self.domain, "source": self.source,
            "rrf_score": round(self.rrf_score, 6), "rank": self.rank,
            "faiss_rank": self.faiss_rank, "graph_rank": self.graph_rank,
            "retrieval_sources": self.retrieval_sources,
        }

    def __repr__(self) -> str:
        return f"RetrievalResult(rank={self.rank}, rrf={self.rrf_score:.4f}, [{'+'.join(self.retrieval_sources)}] '{self.document_name}' p{self.page_number})"


def _rrf_fuse(faiss_results: list, graph_results: list, k: int = RRF_K) -> dict[str, dict]:
    fused: dict[str, dict] = {}

    def _update(results, source_key: str) -> None:
        for r in results:
            cid = r.chunk_id
            if cid not in fused:
                fused[cid] = {"rrf_score": 0.0, "faiss_rank": None, "graph_rank": None, "metadata": r.metadata, "retrieval_sources": []}
            fused[cid]["rrf_score"] += 1.0 / (k + r.rank)
            fused[cid][source_key] = r.rank
            if source_key not in fused[cid]["retrieval_sources"]:
                fused[cid]["retrieval_sources"].append(source_key.replace("_rank", ""))

    _update(faiss_results, "faiss_rank")
    _update(graph_results, "graph_rank")
    return fused


def _resolve_metadata(chunk_id: str, fused_entry: dict, collection=None) -> dict:
    if collection is not None:
        doc = collection.find_one({"_id": chunk_id})
        if doc:
            return doc
    return fused_entry.get("metadata", {})


def retrieve(
    query_text: str,
    faiss_index,
    faiss_chunk_ids: list[str],
    graph,
    embedder,
    collection=None,
    faiss_chunk_metadata: Optional[dict] = None,
    faiss_top_k: int = FAISS_TOP_K,
    graph_top_k: int = GRAPH_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    domain_filter: Optional[str] = None,
    graph_min_coverage: float = 0.3,
) -> list[RetrievalResult]:
    from retrieval.faiss import query as faiss_query
    faiss_results = faiss_query(
        query_text, faiss_index, faiss_chunk_ids, embedder,
        top_k=faiss_top_k, chunk_metadata=faiss_chunk_metadata, collection=collection,
    )

    from retrieval.graph import keyword_search
    # Tokenise: lowercase, strip punctuation, split hyphenated compounds,
    # drop stopwords and tokens shorter than 4 chars.
    raw_tokens: list[str] = []
    for w in query_text.lower().split():
        w = w.strip("?.,!\"'();:")
        # split hyphenated compounds into their parts (e.g. "socio-economically" → ["socio", "economically"])
        for part in w.split("-"):
            part = part.strip("?.,!\"'();:")
            if len(part) > 3 and part not in _STOPWORDS:
                raw_tokens.append(part)
    # deduplicate while preserving order
    keywords = list(dict.fromkeys(raw_tokens))
    graph_results = keyword_search(graph, keywords, top_k=graph_top_k, domain_filter=domain_filter, min_coverage_ratio=graph_min_coverage)

    fused = _rrf_fuse(faiss_results, graph_results)
    sorted_items = sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:final_top_k]

    results: list[RetrievalResult] = []
    for rank, (chunk_id, entry) in enumerate(sorted_items, start=1):
        meta = _resolve_metadata(chunk_id, entry, collection)
        if not meta.get("text") and faiss_chunk_metadata:
            meta = faiss_chunk_metadata.get(chunk_id, meta)
        text = meta.get("text", "") or entry.get("metadata", {}).get("text", "")

        results.append(RetrievalResult(
            chunk_id=chunk_id, text=text,
            document_name=meta.get("document_name", ""), page_number=meta.get("page_number", 0),
            section=meta.get("section"), domain=meta.get("domain", "general"),
            source=meta.get("source", ""), rrf_score=entry["rrf_score"], rank=rank,
            faiss_rank=entry["faiss_rank"], graph_rank=entry["graph_rank"],
            retrieval_sources=entry["retrieval_sources"],
        ))

    logger.info(
        "Extracted FAISS: %d\nExtracted GRAPH: %d\nFinal Extract: %d",
        len(faiss_results), len(graph_results), len(results),
    )

    return results


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     from ingestion.parser import parse_all_pdfs
#     from ingestion.chunker import chunk_documents
#     from ingestion.embedder import Embedder
#     from retrieval.faiss import build_index, load_index
#     from retrieval.graph import build_graph, save_graph, load_graph

#     PDF_DIR   = BASE_DIR / "data" / "pdfs"
#     INDEX_DIR = BASE_DIR / "data" / "indexes" / "faiss"
#     GRAPH_DIR = BASE_DIR / "data" / "indexes" / "graph"

#     print(f"\n{'='*60}")
#     print("HYBRID RETRIEVER — Phase 2 test\n")

#     documents = parse_all_pdfs(PDF_DIR)
#     if not documents:
#         print("No PDFs found.")
#         sys.exit(1)

#     chunks = chunk_documents(documents)
#     embedder = Embedder()
#     embedded = embedder.embed_chunks(chunks)
#     build_index(embedded, INDEX_DIR)
#     G = build_graph(chunks)
#     save_graph(G, GRAPH_DIR)

#     faiss_index, cids, meta_map = load_index(INDEX_DIR)
#     graph = load_graph(GRAPH_DIR)

#     for query_text in [
#         "What is the main goal of the National Education Policy 2020?",
#         "What free services are proposed in public hospitals?",
#     ]:
#         print(f"Query: {query_text}")
#         results = retrieve(
#             query_text=query_text, faiss_index=faiss_index, faiss_chunk_ids=cids,
#             graph=graph, embedder=embedder, faiss_chunk_metadata=meta_map, final_top_k=5,
#         )
#         for r in results:
#             print(f"  [{r.rank}] rrf={r.rrf_score:.5f}  [{'+'.join(r.retrieval_sources)}]  {r.document_name} p{r.page_number}  {r.domain}")
#             print(f"       {r.text[:120]}...")
#         print()

#     print("Hybrid retriever test complete.")
