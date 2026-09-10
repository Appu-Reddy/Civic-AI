from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ai.key import get_gemini_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

class GeneratedResponse(BaseModel):
    answer: str
    evidence_used: list[str]          = Field(default_factory=list)
    is_sufficient: bool               = Field(True)
    insufficiency_note: Optional[str] = Field(None)
    raw_prompt_tokens: int            = Field(0)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer, "evidence_used": self.evidence_used,
            "is_sufficient": self.is_sufficient, "insufficiency_note": self.insufficiency_note,
        }


_SYSTEM_INSTRUCTION = """\
You are DocQA, a precise question-answering assistant that answers questions strictly from the content of provided documents.

STRICT RULES â€” you must follow these without exception:
1. Base your answer ONLY on the evidence provided below. Do not use your training knowledge.
2. Never fabricate facts, figures, dates, names, or claims that are not present in the evidence.
3. If the evidence does not contain enough information to answer, say so clearly and specifically.
4. Always cite the source document and page number for each claim.
5. Use clear, precise language accessible to a general audience.
6. Structure your answer with clear sections when multiple distinct topics are covered.
7. If contextual attributes about the user or query are provided, tailor the answer accordingly.

FORMATTING RULES:
- Output plain text only.
- Do NOT use Markdown formatting.
- Do NOT use bold text or asterisks.
- Never use ** anywhere in the response.
- Do NOT use headings with Markdown symbols such as # or ##.
- Do NOT use Markdown bullet syntax.
- Use simple numbered lists (1., 2., 3.) when listing items.
- Keep citations in plain text, for example: (DocumentName, p10).
- Do not add any formatting characters around important terms.

"""

_ANSWER_PROMPT = """\
User Query: {query}

Query Context:
{user_profile}

{history_context}

{evidence_block}

Based ONLY on the evidence above, provide a clear, helpful answer to the user's query.
If the evidence is insufficient to answer confidently, state this explicitly.
Cite sources as (DocumentName, p{{page}}) after each claim.
"""

_INSUFFICIENT_PROMPT = """\
User Query: {query}

Unfortunately, the documents available do not contain sufficient information to fully answer this query.

Available evidence (partial):
{evidence_block}

Provide a brief response explaining:
1. What information was found (if any)
2. What information is missing
3. Suggest that the user consult the original source documents or other authoritative references for the missing information.

Do not fabricate any details.
"""


def _build_query_context(plan) -> str:
    """Build a compact context block from the generic QueryPlan fields."""
    parts = []
    if plan.domain_hint:          parts.append(f"Domain: {plan.domain_hint}")
    if plan.topic:                parts.append(f"Topic: {plan.topic}")
    if plan.entity_focus:         parts.append(f"Entity of interest: {plan.entity_focus}")
    if plan.context_attributes:   parts.append(f"Context: {', '.join(plan.context_attributes)}")
    return "\n".join(parts) if parts else "Not specified"


def _collect_citations(evidence_list: list) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for ev in evidence_list:
        cite = f"{ev.document_name} p{ev.page_number}"
        if cite not in seen:
            seen.add(cite)
            citations.append(cite)
    return citations


def generate_response(
    query: str,
    plan,
    evidence_list: list,
    history_store,
    min_evidence_threshold: int = 1,
) -> GeneratedResponse:
    from ai.extractor import evidence_to_context_string

    is_sufficient = len(evidence_list) >= min_evidence_threshold

    if is_sufficient:
        history_context = history_store.to_context_string() if history_store else ""
        prompt = _ANSWER_PROMPT.format(
            query=query,
            user_profile=_build_query_context(plan),
            history_context=f"Prior reasoning:\n{history_context}\n" if history_context else "",
            evidence_block=evidence_to_context_string(evidence_list),
        )
    else:
        prompt = _INSUFFICIENT_PROMPT.format(query=query, evidence_block=evidence_to_context_string(evidence_list))

    full_prompt = f"{_SYSTEM_INSTRUCTION}\n\n{prompt}"
    response = get_gemini_model().generate_content(full_prompt)
    answer_text = response.text.strip()

    return GeneratedResponse(
        answer=answer_text,
        evidence_used=_collect_citations(evidence_list),
        is_sufficient=is_sufficient,
        insufficiency_note=(
            "The available documents did not contain sufficient information to fully answer this query."
            if not is_sufficient else None
        ),
        raw_prompt_tokens=len(full_prompt.split()),
    )


## TESTING ##
# if __name__ == "__main__":
#     import sys
#
#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))
#
#     from memory.history import HistoryStore, HistoryEntry, EvidenceRef
#     from ai.planner import QueryPlan
#
#     plan = QueryPlan(
#         raw_query="What are the main findings of the climate study?",
#         domain_hint="climate science", topic="climate findings",
#         keywords=["findings", "climate", "study"], is_multi_hop=False, intent="find_information",
#     )
#     evidence_list = [
#         EvidenceRef("e1", "ClimateReport2023", 5, "Findings", "Global temperatures have risen by 1.1 degrees Celsius since pre-industrial levels.", "ClimateReport2023.pdf"),
#         EvidenceRef("e2", "ClimateReport2023", 8, "Impact", "Sea levels are projected to rise by 0.3 to 1.0 metres by 2100 under high-emission scenarios.", "ClimateReport2023.pdf"),
#     ]
#     history = HistoryStore()
#     history.add(HistoryEntry(1, "Retrieve primary findings", "Two key findings retrieved.", evidence=[evidence_list[0]], satisfied=True))
#
#     print(f"\n{'='*60}")
#     print("GENERATOR — generic test\n")
#     print(f"Query: {plan.raw_query}\nEvidence: {len(evidence_list)} items\n")
#
#     try:
#         resp = generate_response(plan.raw_query, plan, evidence_list, history)
#         print(f"Answer:\n{resp.answer}")
#         print(f"\nSources: {resp.evidence_used}  sufficient={resp.is_sufficient}")
#     except Exception as e:
#         print(f"ERROR: {e}")
#
#     print("\nGenerator test complete.")
