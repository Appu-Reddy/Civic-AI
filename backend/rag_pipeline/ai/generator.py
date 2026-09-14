"""
generator.py — Answer generation

Takes retrieved chunks + sub_query (+ optional prior history for multi-hop),
calls the LLM, and returns a StepResponse with:
    answer   : str            — the generated answer
    evidence : list[str]      — citation strings ("DocName p{N}")
    score    : float          — self-reported confidence 0.0–1.0

Also persists the result to the shared HistoryStore so later steps can
reference prior answers.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ai.key import get_gemini_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


@dataclass
class StepResponse:
    sub_query: str
    answer: str
    evidence: list[str] = field(default_factory=list)   # citation strings
    score: float = 0.0                                   # 0.0 – 1.0
    is_sufficient: bool = True
    insufficiency_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sub_query":          self.sub_query,
            "answer":             self.answer,
            "evidence":           self.evidence,
            "score":              round(self.score, 3),
            "is_sufficient":      self.is_sufficient,
            "insufficiency_note": self.insufficiency_note,
        }


_SYSTEM_INSTRUCTION = """\
You are DocQA, a precise question-answering assistant that answers questions \
strictly from the content of provided documents.

STRICT RULES — follow without exception:
1. Base your answer ONLY on the evidence provided. Do not use outside knowledge.
2. Never fabricate facts, figures, dates, names, or claims absent from the evidence.
3. If evidence is insufficient, say so clearly and set score low.
4. Cite every claim with (DocumentName, pN).
5. Use plain, clear language accessible to a general audience.

FORMATTING RULES:
- Plain text only. No Markdown, no bold, no asterisks, no # headings.
- Numbered lists (1., 2., 3.) when listing items.
- Citations in plain text: (DocumentName, pN).

RESPONSE FORMAT — return ONLY valid JSON with these keys:
{{
  "answer": "<your full plain-text answer with inline citations>",
  "evidence": ["<DocName pN>", ...],
  "score": <float 0.0–1.0 reflecting answer confidence based on evidence quality>
}}
Do NOT wrap in markdown fences.
"""

_ANSWER_PROMPT = """\
Sub-query: {sub_query}

{history_block}\
Retrieved Evidence:
{evidence_block}

Answer the sub-query based ONLY on the evidence above.
{synthesis_hint}\
Return JSON with keys: answer, evidence, score.
"""

_INSUFFICIENT_PROMPT = """\
Sub-query: {sub_query}

{history_block}\
The available documents do not contain sufficient information to answer this sub-query.

Available evidence (partial):
{evidence_block}

Briefly explain what was found and what is missing. Do not fabricate details.
Return JSON with keys: answer, evidence, score (score should be low, e.g. 0.1–0.2).
"""

def _build_evidence_block(chunks: list) -> str:
    """Format retrieved chunks into a numbered evidence block."""
    if not chunks:
        return "No relevant evidence found."
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        section = f" | {chunk.section}" if getattr(chunk, "section", None) else ""
        lines.append(
            f"[{i}] {chunk.document_name} — p{chunk.page_number}{section}\n"
            f"    {chunk.text.strip()}"
        )
    return "\n\n".join(lines)


def _collect_citations(chunks: list) -> list[str]:
    """Deduplicated citation strings from chunk metadata."""
    seen: set[str] = set()
    citations: list[str] = []
    for chunk in chunks:
        cite = f"{chunk.document_name} p{chunk.page_number}"
        if cite not in seen:
            seen.add(cite)
            citations.append(cite)
    return citations


def _clean_json(text: str) -> str:
    """
    Robustly extract and clean a JSON object from raw LLM output.
    1. Normalize escaped formatting markers returned by some LLM responses
    2. Strip markdown fences
    3. Extract outermost {...} block (handles leading/trailing prose)
    """
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ").replace("\\t", " ")

    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    return text


def _parse_llm_json(text: str) -> dict:
    cleaned = _clean_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}\n{cleaned[:300]}")


def generate(
    sub_query: str,
    chunks: list,
    history_store,
    step_number: int,
    is_final_step: bool = False,
    min_chunks: int = 1,
) -> StepResponse:
    """
    Generate an answer for one pipeline step and persist it to history.

    Parameters
    ----------
    sub_query : str
        The question for this step.
    chunks : list[RetrievalResult]
        Retrieved chunks from extractor.extract().
    history_store : HistoryStore
        Shared store; prior answers are read from it and the new answer
        is written back after generation.
    step_number : int
        1-based step index (used when saving to history).
    is_final_step : bool
        When True, the synthesis hint is added to the prompt so the LLM
        knows to compare/summarise all prior step answers.
    min_chunks : int
        Minimum chunks required to attempt a full answer.

    Returns
    -------
    StepResponse
        answer, evidence (citations), score, is_sufficient flag.
    """
    from memory.history import HistoryEntry, EvidenceRef

    is_sufficient = len(chunks) >= min_chunks
    evidence_block = _build_evidence_block(chunks)

    # Build prior-step context from history (empty on step 1)
    history_text = history_store.to_step_context_string()
    history_block = f"{history_text}\n\n" if history_text.strip() else ""

    synthesis_hint = (
        "This is the final step — synthesise and compare the findings from all "
        "prior steps above with the new evidence.\n"
        if is_final_step and history_block
        else ""
    )

    if is_sufficient:
        prompt_body = _ANSWER_PROMPT.format(
            sub_query=sub_query,
            history_block=history_block,
            evidence_block=evidence_block,
            synthesis_hint=synthesis_hint,
        )
    else:
        prompt_body = _INSUFFICIENT_PROMPT.format(
            sub_query=sub_query,
            history_block=history_block,
            evidence_block=evidence_block,
        )

    full_prompt = f"{_SYSTEM_INSTRUCTION}\n\n{prompt_body}"

    try:
        response = get_gemini_model().generate_content(full_prompt)
        data = _parse_llm_json(response.text.strip())

        answer = str(data.get("answer", "")).strip()
        raw_evidence = data.get("evidence", [])
        score = float(data.get("score", 0.5))

        if not answer:
            raise ValueError("LLM returned an empty answer field")

        # Merge LLM-reported citations with chunk-derived citations
        chunk_citations = _collect_citations(chunks)
        llm_citations = [str(e) for e in raw_evidence if isinstance(e, str)]
        merged_evidence = list(dict.fromkeys(chunk_citations + llm_citations))

        step_response = StepResponse(
            sub_query=sub_query,
            answer=answer,
            evidence=merged_evidence,
            score=min(max(score, 0.0), 1.0),
            is_sufficient=is_sufficient,
            insufficiency_note=(
                "Insufficient evidence found for this step." if not is_sufficient else None
            ),
        )

    except Exception as exc:
        logger.warning("generate step %d failed (%s) — using fallback.", step_number, exc)
        step_response = StepResponse(
            sub_query=sub_query,
            answer="Unable to generate an answer for this step due to an internal error.",
            evidence=_collect_citations(chunks),
            score=0.0,
            is_sufficient=False,
            insufficiency_note=str(exc),
        )

    # ── Persist to history ────────────────────────────────────────────────────
    evidence_refs = [
        EvidenceRef(
            chunk_id=c.chunk_id,
            document_name=c.document_name,
            page_number=c.page_number,
            section=getattr(c, "section", None),
            text=c.text,
            source=getattr(c, "source", ""),
        )
        for c in chunks
    ]

    history_store.add(HistoryEntry(
        step_number=step_number,
        objective=sub_query,
        summary=f"{len(chunks)} chunk(s) retrieved." if chunks else "No evidence found.",
        evidence=evidence_refs,
        entities={},
        satisfied=is_sufficient,
        sub_query=sub_query,
        answer=step_response.answer,
    ))

    logger.info("Generating...")
    return step_response
