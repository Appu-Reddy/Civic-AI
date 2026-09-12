"""
extractor.py — Retrieval layer

Takes a sub_query and a list of domain slugs, runs hybrid retrieval
(FAISS semantic search + graph keyword search via hybrid.py), and
returns the fused RetrievalResult chunks.

Public API
----------
extract(sub_query, domains, retrieval_ctx, final_top_k) -> list[RetrievalResult]
"""

import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)


def extract(
    sub_query: str,
    domains: list[str],
    retrieval_ctx,
    final_top_k: int = 5,
) -> list:
    """
    Run hybrid retrieval for a single step.

    Parameters
    ----------
    sub_query : str
        The step question used as the FAISS semantic query.
    domains : list[str]
        Domain slugs from step_definer (e.g. ["india_nep"]).
        The first entry is passed as domain_filter to graph retrieval.
        Empty list means no domain filtering.
    retrieval_ctx : RetrievalContext
        Holds faiss_index, faiss_chunk_ids, faiss_chunk_metadata,
        graph, embedder, and optional mongo collection.
    final_top_k : int
        Maximum number of fused results to return.

    Returns
    -------
    list[RetrievalResult]
        Fused, ranked retrieval results ready for the generator.
    """
    from retrieval.hybrid import retrieve

    domain_filter = domains[0] if domains else None
    results = retrieve(
        query_text=sub_query,
        faiss_index=retrieval_ctx.faiss_index,
        faiss_chunk_ids=retrieval_ctx.faiss_chunk_ids,
        graph=retrieval_ctx.graph,
        embedder=retrieval_ctx.embedder,
        collection=retrieval_ctx.collection,
        faiss_chunk_metadata=retrieval_ctx.faiss_chunk_metadata,
        final_top_k=final_top_k,
        domain_filter=domain_filter,
    )

    logger.info("Extracted: %d chunks", len(results))
    return results