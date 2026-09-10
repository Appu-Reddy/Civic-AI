from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MAX_STEPS: int = 3


@dataclass
class RetrievalStep:
    step_number: int
    objective: str
    retrieval_query: str
    keywords: list[str] = field(default_factory=list)
    domain_filter: Optional[str] = None
    depends_on: list[int] = field(default_factory=list)
    is_final: bool = False

    def __repr__(self) -> str:
        return f"RetrievalStep({self.step_number}: '{self.objective[:50]}', final={self.is_final})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_keywords(plan) -> list[str]:
    """Return the plan's keywords, deduplicated and lowercased."""
    return list(dict.fromkeys(kw.lower() for kw in plan.keywords if kw))


def _context_suffix(plan) -> str:
    """Build a compact context string from any non-null plan attributes."""
    parts = []
    if plan.entity_focus:       parts.append(plan.entity_focus)
    if plan.topic:              parts.append(plan.topic)
    if plan.domain_hint:        parts.append(plan.domain_hint)
    if plan.context_attributes: parts.extend(plan.context_attributes)
    return " ".join(parts).strip()


def _domain_filter(plan) -> Optional[str]:
    """
    Domain filter is intentionally disabled at the step level.

    Chunks are tagged with the document filename slug (e.g. "india_nep"),
    not a subject-matter label (e.g. "education").  The LLM's domain_hint
    is a topic label, not a document identity, so the two namespaces never
    match — applying a filter here would silently exclude all results.

    Domain-awareness is provided by the retrieval query itself (which
    includes topic, entity_focus, and keywords) together with FAISS cosine
    similarity and graph keyword scoring.
    """
    return None


# ---------------------------------------------------------------------------
# Intent templates
# ---------------------------------------------------------------------------

def _steps_find_information(plan) -> list[RetrievalStep]:
    """Broad retrieval: find relevant information on the topic."""
    ctx = _context_suffix(plan)
    kw = _base_keywords(plan)
    return [
        RetrievalStep(
            step_number=1,
            objective=f"Retrieve primary information about: {plan.topic or plan.raw_query}",
            retrieval_query=f"{ctx} {plan.raw_query}".strip(),
            keywords=kw,
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=False,
        ),
        RetrievalStep(
            step_number=2,
            objective="Retrieve supporting details, context, and related specifics",
            retrieval_query=f"details context {ctx}".strip(),
            keywords=kw + ["details", "context", "background"],
            domain_filter=_domain_filter(plan),
            depends_on=[1], is_final=True,
        ),
    ]


def _steps_explain_concept(plan) -> list[RetrievalStep]:
    """Single-step: retrieve the definition and explanation of a concept."""
    ctx = _context_suffix(plan)
    kw = _base_keywords(plan)
    return [
        RetrievalStep(
            step_number=1,
            objective=f"Retrieve definition and explanation of: {plan.entity_focus or plan.topic or plan.raw_query}",
            retrieval_query=f"definition explanation {ctx} {plan.raw_query}".strip(),
            keywords=kw + ["definition", "explanation", "meaning", "refers to"],
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=True,
        ),
    ]


def _steps_compare_items(plan) -> list[RetrievalStep]:
    """Three-step: retrieve details on each item, then compare."""
    ctx = _context_suffix(plan)
    kw = _base_keywords(plan)
    kw_a = kw[:3] if len(kw) >= 1 else kw
    kw_b = kw[1:4] if len(kw) >= 2 else kw
    return [
        RetrievalStep(
            step_number=1,
            objective=f"Retrieve details for the first item: {kw[0] if kw else ctx}",
            retrieval_query=f"{kw[0] if kw else ctx} details description".strip(),
            keywords=kw_a,
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=False,
        ),
        RetrievalStep(
            step_number=2,
            objective=f"Retrieve details for the second item: {kw[1] if len(kw) > 1 else ctx}",
            retrieval_query=f"{kw[1] if len(kw) > 1 else ctx} details description".strip(),
            keywords=kw_b,
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=False,
        ),
        RetrievalStep(
            step_number=3,
            objective="Compare the items: differences, similarities, and key distinctions",
            retrieval_query=f"comparison difference between {ctx}".strip(),
            keywords=["comparison", "difference", "versus", "contrast", "similarity"],
            domain_filter=_domain_filter(plan),
            depends_on=[1, 2], is_final=True,
        ),
    ]


def _steps_find_details(plan) -> list[RetrievalStep]:
    """Single-step: retrieve specific details (numbers, dates, specs, etc.)."""
    ctx = _context_suffix(plan)
    kw = _base_keywords(plan)
    return [
        RetrievalStep(
            step_number=1,
            objective=f"Retrieve specific details for: {plan.entity_focus or plan.topic or plan.raw_query}",
            retrieval_query=f"specific details {ctx} {plan.raw_query}".strip(),
            keywords=kw + ["specific", "details", "data", "figure", "specification"],
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=True,
        ),
    ]


def _steps_summarise_topic(plan) -> list[RetrievalStep]:
    """Two-step: retrieve broad overview then key supporting points."""
    ctx = _context_suffix(plan)
    kw = _base_keywords(plan)
    return [
        RetrievalStep(
            step_number=1,
            objective=f"Retrieve overview and main points about: {plan.topic or plan.raw_query}",
            retrieval_query=f"overview summary {ctx}".strip(),
            keywords=kw + ["overview", "summary", "introduction", "main"],
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=False,
        ),
        RetrievalStep(
            step_number=2,
            objective="Retrieve key supporting evidence, examples, and conclusions",
            retrieval_query=f"examples evidence conclusions {ctx}".strip(),
            keywords=kw + ["example", "evidence", "conclusion", "result", "finding"],
            domain_filter=_domain_filter(plan),
            depends_on=[1], is_final=True,
        ),
    ]


def _steps_trace_relationship(plan) -> list[RetrievalStep]:
    """Three-step: identify entities, retrieve how they interact, then synthesise."""
    ctx = _context_suffix(plan)
    kw = _base_keywords(plan)
    return [
        RetrievalStep(
            step_number=1,
            objective=f"Identify the primary entities involved: {ctx or plan.raw_query}",
            retrieval_query=f"identify entities {ctx} {plan.raw_query}".strip(),
            keywords=kw,
            domain_filter=_domain_filter(plan),
            depends_on=[], is_final=False,
        ),
        RetrievalStep(
            step_number=2,
            objective="Retrieve how the entities interact, depend on, or affect each other",
            retrieval_query=f"relationship interaction dependency {ctx}".strip(),
            keywords=kw + ["relationship", "interaction", "depends on", "causes", "linked to"],
            domain_filter=_domain_filter(plan),
            depends_on=[1], is_final=False,
        ),
        RetrievalStep(
            step_number=3,
            objective="Retrieve consequences, outcomes, or conclusions of the relationship",
            retrieval_query=f"outcome consequence result {ctx}".strip(),
            keywords=kw + ["outcome", "consequence", "result", "therefore", "leads to"],
            domain_filter=_domain_filter(plan),
            depends_on=[1, 2], is_final=True,
        ),
    ]


def _steps_general_info(plan) -> list[RetrievalStep]:
    """Fallback: single-step retrieval using the raw query directly."""
    return [RetrievalStep(
        step_number=1,
        objective="Retrieve information relevant to the query",
        retrieval_query=plan.raw_query,
        keywords=_base_keywords(plan),
        domain_filter=_domain_filter(plan),
        depends_on=[], is_final=True,
    )]


# ---------------------------------------------------------------------------
# Intent dispatch map
# ---------------------------------------------------------------------------

_INTENT_MAP = {
    "find_information":   _steps_find_information,
    "explain_concept":    _steps_explain_concept,
    "compare_items":      _steps_compare_items,
    "find_details":       _steps_find_details,
    "summarise_topic":    _steps_summarise_topic,
    "trace_relationship": _steps_trace_relationship,
    "general_info":       _steps_general_info,
}


def define_steps(plan) -> list[RetrievalStep]:
    intent = plan.intent or "general_info"
    steps = _INTENT_MAP.get(intent, _steps_general_info)(plan)

    # Collapse multi-step flows for simple (non-multi-hop) queries:
    # keep independent steps + the final step, renumber, clear depends_on.
    if not plan.is_multi_hop and len(steps) > 1:
        final_steps = [s for s in steps if s.is_final]
        independent_steps = [s for s in steps if not s.depends_on]
        kept = {s.step_number: s for s in (independent_steps + final_steps)}
        steps = list(kept.values())
        for i, s in enumerate(sorted(steps, key=lambda x: x.step_number), start=1):
            s.step_number = i
            s.depends_on = []

    # Hard cap at MAX_STEPS
    if len(steps) > MAX_STEPS:
        non_final = [s for s in steps if not s.is_final]
        final = [s for s in steps if s.is_final]
        kept = (non_final[: MAX_STEPS - 1] + final[:1]) if final else steps[:MAX_STEPS]
        steps = []
        for i, s in enumerate(sorted(kept, key=lambda x: x.step_number), start=1):
            s.step_number = i
            s.is_final = (i == len(kept))
            steps.append(s)

    return steps


## TESTING ##
# if __name__ == "__main__":
#     import sys
#
#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))
#
#     from ai.planner import plan_query
#
#     print(f"\n{'='*60}")
#     print("STEP DEFINER — generic test\n")
#
#     for query in [
#         "What is the main argument of the document about climate policy?",
#         "Compare the two approaches to carbon pricing described in the report.",
#         "What are the key findings of the health study?",
#     ]:
#         print(f"Query: {query}")
#         try:
#             plan = plan_query(query)
#             steps = define_steps(plan)
#             print(f"  intent={plan.intent}  multi_hop={plan.is_multi_hop}  steps={len(steps)}")
#             for s in steps:
#                 print(f"    Step {s.step_number}: {s.objective}")
#                 print(f"      query   : {s.retrieval_query[:80]}")
#                 print(f"      keywords: {s.keywords}  final={s.is_final}")
#         except Exception as e:
#             print(f"  ERROR: {e}")
#         print()
#
#     print("Step-Definer test complete.")
