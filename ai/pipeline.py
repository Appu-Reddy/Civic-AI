import logging
import re as _re
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
    objective: str
    evidence_count: int
    evidence_categories: list[str]
    sources: list[str]
    satisfied: bool
    skipped: bool = False
    skip_reason: str = ""


class PipelineResponse(BaseModel):
    query: str
    answer: str
    is_sufficient: bool
    is_valid: bool
    grounding_score: float
    citations: list[str]             = Field(default_factory=list)
    steps_executed: list[StepSummary]= Field(default_factory=list)
    flagged_sentences: list[str]     = Field(default_factory=list)
    validation_note: str             = ""
    insufficiency_note: Optional[str]= None
    elapsed_seconds: float           = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query, "answer": self.answer,
            "is_sufficient": self.is_sufficient, "is_valid": self.is_valid,
            "grounding_score": round(self.grounding_score, 3), "citations": self.citations,
            "steps_executed": [s.model_dump() for s in self.steps_executed],
            "flagged_sentences": self.flagged_sentences, "validation_note": self.validation_note,
            "insufficiency_note": self.insufficiency_note, "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class RetrievalContext:
    faiss_index: object
    faiss_chunk_ids: list[str]
    faiss_chunk_metadata: dict
    graph: object
    embedder: object
    collection: Optional[object] = None


def _enrich_step_from_history(step, history) -> None:
    """
    Mutate step.retrieval_query and step.keywords in-place by injecting
    entity names discovered in the history entries this step depends on.

    Pulls from HistoryEntry.entities, which is keyed by evidence category.
    We extract short, meaningful tokens from entity values and prepend them
    to the retrieval query so FAISS and graph searches in this step are
    anchored to what was actually found in prior steps rather than the
    original static template query.
    """
    if not step.depends_on:
        return

    injected_terms: list[str] = []

    for dep_step_num in step.depends_on:
        entry = history.get(dep_step_num)
        if entry is None or not entry.satisfied:
            continue

        # Collect entity values from history entries — keep only short,
        # meaningful tokens (not full sentences).
        for entity_type, values in entry.entities.items():
            for val in values:
                clean = val.strip(" .,;:\"'").split("\n")[0]
                if 3 < len(clean) <= 80 and clean not in injected_terms:
                    injected_terms.append(clean)

        # Also extract capitalised multi-word proper nouns from prior evidence
        # text — these are the domain-agnostic named entities most likely to
        # appear verbatim in related chunks.
        for ev in entry.evidence:
            for m in _re.finditer(
                r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\b",
                ev.text,
            ):
                name = m.group(1).strip()
                if len(name) <= 80 and name not in injected_terms:
                    injected_terms.append(name)

    if not injected_terms:
        return

    # Prepend the discovered terms to the retrieval query so the semantic
    # search is steered toward what was found in prior steps.
    prefix = " ".join(injected_terms[:6])   # cap at 6 terms to avoid noise
    step.retrieval_query = f"{prefix} {step.retrieval_query}".strip()

    # Add as keywords for the graph keyword search too
    for term in injected_terms[:6]:
        words = [w.lower() for w in term.split() if len(w) > 3]
        for w in words:
            if w not in step.keywords:
                step.keywords.append(w)

def run_pipeline(    
    query: str,
    retrieval_ctx: RetrievalContext,
    final_top_k: int = 5,
    grounding_threshold: float = 0.6,
) -> PipelineResponse:
    from ai.planner import plan_query
    from ai.step_definer import define_steps
    from ai.extractor import extract_evidence
    from ai.generator import generate_response
    from ai.validator import validate_response, format_validated_response
    from memory.history import HistoryStore, HistoryEntry
    from retrieval.hybrid import retrieve

    wall_start = time.time()
    history = HistoryStore()
    steps_executed: list[StepSummary] = []
    all_evidence_flat: list = []

    logger.info("Planning...")
    plan = plan_query(query)
    steps = define_steps(plan)

    logger.info("Retrieving...")
    for step in steps:
        if history.covers_objective(step.objective):
            steps_executed.append(StepSummary(
                step_number=step.step_number, objective=step.objective,
                evidence_count=0, evidence_categories=[], sources=[],
                satisfied=True, skipped=True, skip_reason="covered by prior step",
            ))
            continue

        # Enrich this step's query with entities found in its dependency steps
        _enrich_step_from_history(step, history)

        raw_results = retrieve(
            query_text=step.retrieval_query,
            faiss_index=retrieval_ctx.faiss_index, faiss_chunk_ids=retrieval_ctx.faiss_chunk_ids,
            graph=retrieval_ctx.graph, embedder=retrieval_ctx.embedder,
            collection=retrieval_ctx.collection, faiss_chunk_metadata=retrieval_ctx.faiss_chunk_metadata,
            final_top_k=final_top_k, domain_filter=step.domain_filter,
        )

        evidence = extract_evidence(
            retrieval_results=raw_results, step_objective=step.objective,
            step_keywords=step.keywords, step_domain=step.domain_filter,
        )

        seen_ids = {e.chunk_id for e in all_evidence_flat}
        for ev in evidence:
            if ev.chunk_id not in seen_ids:
                all_evidence_flat.append(ev)
                seen_ids.add(ev.chunk_id)

        entities: dict[str, list[str]] = {}
        for ev in evidence:
            cat = ev.__dict__.get("category", "general")
            entities.setdefault(cat, []).append(ev.text[:60].strip())
            # Extract capitalised multi-word proper nouns from evidence text so
            # dependent steps can inject them into their retrieval queries.
            for m in _re.finditer(
                r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\b",
                ev.text,
            ):
                entity_name = m.group(1).strip()
                if len(entity_name) <= 80:
                    entities.setdefault("named_entity", [])
                    if entity_name not in entities["named_entity"]:
                        entities["named_entity"].append(entity_name)

        history.add(HistoryEntry(
            step_number=step.step_number, objective=step.objective,
            summary=f"Retrieved {len(evidence)} item(s)" if evidence else "No evidence found.",
            evidence=evidence, entities=entities, satisfied=len(evidence) > 0,
        ))

        categories = list({ev.__dict__.get("category", "general") for ev in evidence})
        sources = list({f"{ev.document_name} p{ev.page_number}" for ev in evidence})
        steps_executed.append(StepSummary(
            step_number=step.step_number, objective=step.objective,
            evidence_count=len(evidence), evidence_categories=categories,
            sources=sources, satisfied=len(evidence) > 0,
        ))

    logger.info("Generating...")
    generated = generate_response(query=query, plan=plan, evidence_list=all_evidence_flat, history_store=history)
    logger.info("Validating Response...")
    validation = validate_response(generated_response=generated, evidence_list=all_evidence_flat, grounding_threshold=grounding_threshold)
    final_answer = format_validated_response(generated, validation)

    elapsed = round(time.time() - wall_start, 2)
    logger.info("Final Response Ready.")

    return PipelineResponse(
        query=query, answer=final_answer,
        is_sufficient=generated.is_sufficient, is_valid=validation.is_valid,
        grounding_score=validation.grounding_score,
        citations=sorted({f"{ev.document_name} p{ev.page_number}" for ev in all_evidence_flat}),
        steps_executed=steps_executed, flagged_sentences=validation.flagged_sentences,
        validation_note=validation.validation_note, insufficiency_note=generated.insufficiency_note,
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
    graph = load_graph(graph_dir)
    embedder = Embedder(model_name=model_name)

    logger.info("Embeddings loaded.")
    return RetrievalContext(
        faiss_index=faiss_index, faiss_chunk_ids=faiss_chunk_ids,
        faiss_chunk_metadata=faiss_chunk_metadata, graph=graph,
        embedder=embedder, collection=mongo_collection,
    )


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     INDEX_DIR = BASE_DIR / "data" / "indexes" / "faiss"
#     GRAPH_DIR = BASE_DIR / "data" / "indexes" / "graph"

#     print(f"\n{'='*60}")
#     print("AI PIPELINE — Phase 3 end-to-end test")
#     print(f"FAISS: {INDEX_DIR}\nGraph: {GRAPH_DIR}\n")

#     if not (INDEX_DIR / "civic_ai.index").exists():
#         print("ERROR: FAISS index not found. Run: python -m ingestion.ingest --skip-mongo")
#         sys.exit(1)
#     if not (GRAPH_DIR / "civic_ai_graph.pkl").exists():
#         print("ERROR: Graph not found. Run: python -m ingestion.ingest --skip-mongo")
#         sys.exit(1)

#     try:
#         ctx = load_retrieval_context(INDEX_DIR, GRAPH_DIR)
#         print(f"FAISS: {ctx.faiss_index.ntotal} vectors  Graph: {ctx.graph.number_of_nodes()} nodes\n")
#     except Exception as e:
#         print(f"ERROR loading context: {e}")
#         sys.exit(1)

#     for query in [
#         "What are the main objectives described in the document?",
#         "Summarise the key findings of the report.",
#     ]:
#         print(f"Query: {query}")
#         try:
#             response = run_pipeline(query, ctx)
#             print(f"\nAnswer:\n{response.answer}")
#             print(f"\nCitations: {response.citations}")
#             print(f"Valid={response.is_valid}  Grounding={response.grounding_score:.2f}  Elapsed={response.elapsed_seconds}s")
#             for s in response.steps_executed:
#                 skip = " [SKIPPED]" if s.skipped else ""
#                 print(f"  Step {s.step_number}: {s.objective}{skip}  evidence={s.evidence_count}")
#             if response.flagged_sentences:
#                 print(f"Flagged ({len(response.flagged_sentences)}): {response.flagged_sentences[0][:100]}")
#         except Exception as e:
#             import traceback
#             print(f"ERROR: {e}")
#             traceback.print_exc()
#         print()

#     print("Pipeline test complete.")