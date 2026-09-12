"""
pipeline.py — Orchestrator

Wires planner → step_definer → extractor → generator → validator into
two clean workflows:

Single-hop
----------
  planner → step_definer → extractor → generator → validator → answer

Multi-hop
---------
  planner → for each step:
                step_definer → extractor → generator (saves to history)
            → validator on final step's answer → answer

Public API
----------
  run_pipeline(query, retrieval_ctx, final_top_k, grounding_threshold)
      -> PipelineResponse

  load_retrieval_context(...) -> RetrievalContext
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


class StepSummary(BaseModel):
    step_number: int
    sub_query: str       = ""
    answer: str          = ""
    evidence: list[str]  = Field(default_factory=list)
    score: float         = 0.0
    is_sufficient: bool  = True
    is_final: bool       = False


class PipelineResponse(BaseModel):
    query: str
    is_multi_hop: bool               = False
    answer: str                      = ""
    is_sufficient: bool              = True
    is_valid: bool                   = True
    grounding_score: float           = 1.0
    quality_score: float             = 1.0
    offensive_score: float           = 0.0
    citations: list[str]             = Field(default_factory=list)
    steps: list[StepSummary]         = Field(default_factory=list)
    flagged_sentences: list[str]     = Field(default_factory=list)
    validation_note: str             = ""
    elapsed_seconds: float           = 0.0

    def to_dict(self) -> dict:
        return {
            "query":             self.query,
            "is_multi_hop":      self.is_multi_hop,
            "answer":            self.answer,
            "is_sufficient":     self.is_sufficient,
            "is_valid":          self.is_valid,
            "grounding_score":   round(self.grounding_score, 3),
            "quality_score":     round(self.quality_score, 3),
            "offensive_score":   round(self.offensive_score, 3),
            "citations":         self.citations,
            "steps":             [s.model_dump() for s in self.steps],
            "flagged_sentences": self.flagged_sentences,
            "validation_note":   self.validation_note,
            "elapsed_seconds":   round(self.elapsed_seconds, 2),
        }


@dataclass
class RetrievalContext:
    faiss_index: object
    faiss_chunk_ids: list
    faiss_chunk_metadata: dict
    graph: object
    embedder: object
    collection: Optional[object] = None


def run_pipeline(
    query: str,
    retrieval_ctx: RetrievalContext,
    final_top_k: int = 10,
    grounding_threshold: float = 0.5,
) -> PipelineResponse:
    """
    Entry point for all queries.

    1. planner.plan_query()         — 1 LLM call, returns is_multi_hop + steps
    2. step_definer.define_steps()  — no LLM, adds domains to each step
    3. For each step (sequential):
         a. extractor.extract()     — FAISS + graph retrieval
         b. generator.generate()    — 1 LLM call, saves answer to history
    4. validator.validate_response() — grounding check on final answer

    Total LLM calls: 1 (plan) + N (generate, N = number of steps, max 3) = max 4.
    """
    from ai.planner import plan_query
    from ai.step_definer import define_steps
    from ai.extractor import extract
    from ai.generator import generate
    from ai.validator import validate, format_answer
    from memory.history import HistoryStore

    wall_start = time.time()

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    logger.info("Planning query...")
    plan = plan_query(query)
    logger.info(
        "Pipeline Plan: is_multi_hop=%s  steps=%d",
        plan.is_multi_hop, len(plan.steps),
    )

    # ── Step 2: Enrich steps with domains ────────────────────────────────────
    enriched_steps = define_steps(plan.steps)
    total_steps = len(enriched_steps)

    # ── Step 3: Execute each step ────────────────────────────────────────────
    history = HistoryStore()
    step_summaries: list[StepSummary] = []
    all_chunks: list = []          # deduplicated across steps for validation

    for i, step in enumerate(enriched_steps, start=1):
        sub_query = step["sub_query"]
        domains   = step.get("domains", [])
        is_final  = (i == total_steps)

        logger.info("\nPipeline: step %d/%d  sub_query=%r", i, total_steps, sub_query[:60])

        # a. Extract
        chunks = extract(
            sub_query=sub_query,
            domains=domains,
            retrieval_ctx=retrieval_ctx,
            final_top_k=final_top_k,
        )

        # Accumulate unique chunks for final validation
        seen_ids = {c.chunk_id for c in all_chunks}
        for c in chunks:
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)

        # b. Generate (writes to history internally)
        step_resp = generate(
            sub_query=sub_query,
            chunks=chunks,
            history_store=history,
            step_number=i,
            is_final_step=is_final and plan.is_multi_hop,
        )

        step_summaries.append(StepSummary(
            step_number=i,
            sub_query=sub_query,
            answer=step_resp.answer,
            evidence=step_resp.evidence,
            score=step_resp.score,
            is_sufficient=step_resp.is_sufficient,
            is_final=is_final,
        ))

    # ── Step 4: Validate final answer ────────────────────────────────────────
    final_step = step_summaries[-1] if step_summaries else None

    if final_step is None:
        elapsed = round(time.time() - wall_start, 2)
        return PipelineResponse(
            query=query, is_multi_hop=plan.is_multi_hop,
            answer="No steps were executed.",
            is_sufficient=False, is_valid=False, grounding_score=0.0,
            elapsed_seconds=elapsed,
        )

    # Build a minimal GeneratedResponse-compatible object for the validator
    class _GR:
        def __init__(self, answer, is_sufficient, insufficiency_note=None):
            self.answer = answer
            self.is_sufficient = is_sufficient
            self.insufficiency_note = insufficiency_note

    gr = _GR(
        answer=final_step.answer,
        is_sufficient=final_step.is_sufficient,
        insufficiency_note=None if final_step.is_sufficient else "Insufficient evidence.",
    )

    validation   = validate(query, gr.answer, all_chunks, grounding_threshold)
    final_answer = format_answer(gr.answer, validation)

    # Collect all citations across all steps
    all_citations = sorted({cite for s in step_summaries for cite in s.evidence})

    elapsed = round(time.time() - wall_start, 2)

    logger.info("Final Response Ready.")

    return PipelineResponse(
        query=query,
        is_multi_hop=plan.is_multi_hop,
        answer=final_answer,
        is_sufficient=final_step.is_sufficient,
        is_valid=validation.is_valid,
        grounding_score=validation.grounding_score,
        quality_score=validation.quality_score,
        offensive_score=validation.offensive_score,
        citations=all_citations,
        steps=step_summaries,
        flagged_sentences=validation.flagged_sentences,
        validation_note=validation.validation_note,
        elapsed_seconds=elapsed,
    )


def load_retrieval_context(
    index_dir: Optional[Path] = None,
    graph_dir: Optional[Path] = None,
    model_name: str = "all-MiniLM-L6-v2",
    mongo_collection=None,
) -> RetrievalContext:
    from retrieval.faiss import load_index, DEFAULT_INDEX_DIR
    from retrieval.graph import load_graph, DEFAULT_GRAPH_DIR
    from ingestion.embedder import Embedder

    index_dir = index_dir or DEFAULT_INDEX_DIR
    graph_dir = graph_dir or DEFAULT_GRAPH_DIR

    faiss_index, faiss_chunk_ids, faiss_chunk_metadata = load_index(index_dir)
    graph   = load_graph(graph_dir)
    embedder = Embedder(model_name=model_name)

    logger.info("Embeddings Loaded.")
    return RetrievalContext(
        faiss_index=faiss_index,
        faiss_chunk_ids=faiss_chunk_ids,
        faiss_chunk_metadata=faiss_chunk_metadata,
        graph=graph,
        embedder=embedder,
        collection=mongo_collection,
    )
