# -*- coding: utf-8 -*-
"""
validator.py -- Response validation

Uses the LLM (via key.py rotation) to evaluate the final answer against
the original query and retrieved evidence on three axes:

    grounding_score  : float 0-1   how well the answer is supported by evidence
    quality_score    : float 0-1   overall answer quality / relevance to query
    offensive_score  : float 0-1   likelihood of harmful / inappropriate content
                                   (0 = clean, 1 = definitely offensive)

A response is valid when:
    grounding_score  >= grounding_threshold  (default 0.6)
    offensive_score  <  offensive_threshold  (default 0.5)

Falls back to a heuristic check if the LLM call fails.

Public API
----------
    validate(query, answer, chunks, grounding_threshold, offensive_threshold)
        -> ValidationResult

    format_answer(answer, validation)
        -> str
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from ai.key import get_gemini_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    grounding_score: float
    quality_score: float
    offensive_score: float
    flagged_sentences: list = field(default_factory=list)
    validation_note: str = ""

    def to_dict(self) -> dict:
        return {
            "is_valid":          self.is_valid,
            "grounding_score":   round(self.grounding_score, 3),
            "quality_score":     round(self.quality_score, 3),
            "offensive_score":   round(self.offensive_score, 3),
            "flagged_sentences": self.flagged_sentences,
            "validation_note":   self.validation_note,
        }


_VALIDATOR_SYSTEM = (
    "You are a fair and practical quality-control assistant for a document "
    "question-answering system.\n\n"
    "You will be given:\n"
    "1. The original user query\n"
    "2. The generated answer\n"
    "3. A ranked sample of the retrieved evidence chunks used to produce the answer\n\n"
    "Evaluate the generated answer on THREE dimensions.\n\n"
    "CRITICAL: The evidence shown is only a top-ranked SAMPLE. The answer may have been "
    "generated using additional chunks not shown here. Therefore:\n"
    "- Do NOT penalise a claim simply because the exact supporting text is absent from "
    "the evidence sample below.\n"
    "- Only flag a claim as ungrounded if it clearly contradicts the evidence, or if it "
    "introduces a specific fact (number, date, name, policy detail) that is nowhere "
    "plausibly traceable to ANY of the evidence chunks shown.\n"
    "- Reasonable paraphrasing, summarising, and synthesising across chunks is GOOD.\n"
    "- A well-cited answer that covers many points deserves a HIGH grounding score even "
    "if not every individual sentence has a verbatim match in the sample.\n"
    "- Do NOT penalise concise answers or minor omissions.\n"
    "- Expressing uncertainty when evidence is thin is acceptable.\n\n"
    "grounding_score (0.0 to 1.0)\n"
    "How well is the answer supported by the evidence, given that only a sample is shown?\n"
    "  0.9-1.0 : All claims are clearly traceable to evidence (direct or paraphrased). "
    "No contradictions.\n"
    "  0.7-0.8 : Most claims supported; minor inferences that do not contradict evidence.\n"
    "  0.5-0.6 : About half supported; some meaningful unsupported or contradictory claims.\n"
    "  0.3-0.4 : Several important claims contradict or are entirely absent from evidence.\n"
    "  0.0-0.2 : Answer is mostly fabricated or directly contradicts the evidence.\n\n"
    "quality_score (0.0 to 1.0)\n"
    "How useful and relevant is the answer to the query?\n"
    "  0.9-1.0 : Directly and completely answers the query in clear language.\n"
    "  0.7-0.8 : Answers the main question; minor gaps or slight verbosity.\n"
    "  0.5-0.6 : Partially answers; missing one or two notable points.\n"
    "  0.3-0.4 : Significantly incomplete, unclear, or poorly aligned.\n"
    "  0.0-0.2 : Irrelevant, incoherent, or fails to address the query.\n\n"
    "offensive_score (0.0 to 1.0)\n"
    "Is there harmful, biased, or inappropriate content in the answer?\n"
    "  0.0     : Completely clean -- the expected default for factual document answers.\n"
    "  0.1-0.2 : Mildly awkward phrasing, no real harm.\n"
    "  0.3-0.5 : Borderline or potentially insensitive language.\n"
    "  0.6-0.8 : Clearly inappropriate, biased, or derogatory.\n"
    "  0.9-1.0 : Explicitly hateful, abusive, or dangerous.\n"
    "  NOTE: Do NOT flag neutral discussion of sensitive topics.\n\n"
    "flagged_sentences\n"
    "  Copy ONLY sentences that clearly CONTRADICT the evidence or introduce a specific "
    "fabricated fact absent from all chunks. Do NOT flag paraphrases or inferences.\n"
    "  Return an empty list for a good, well-cited answer -- this is the expected result.\n\n"
    "note\n"
    "  One-line plain-text summary.\n\n"
    "Return ONLY valid JSON with keys: grounding_score, quality_score, offensive_score, "
    "flagged_sentences, note.\n"
    "Do not wrap in markdown. Do not add any text outside the JSON object.\n"
)

_VALIDATOR_PROMPT = (
    "Query:\n{query}\n\n"
    "Generated Answer:\n{answer}\n\n"
    "Retrieved Evidence:\n{evidence_block}\n"
)


def _build_evidence_block(chunks: list) -> str:
    if not chunks:
        return "No evidence provided."
    lines = []
    for i, c in enumerate(chunks, start=1):
        section = " | " + c.section if getattr(c, "section", None) else ""
        lines.append(
            "[" + str(i) + "] " + c.document_name + " p" + str(c.page_number) + section + "\n"
            "    " + c.text.strip()[:300]
        )
    return "\n\n".join(lines)


def _clean_json(text: str) -> str:
    # Some LLM responses contain literal escaped line breaks between JSON
    # tokens rather than actual formatting whitespace.
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ").replace("\\t", " ")

    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    return text


def _parse_json(text: str) -> dict:
    cleaned = _clean_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON from validator response: {e}\n{cleaned[:300]}")


def _clamp(value, lo: float = 0.0, hi: float = 1.0) -> float:
    v = float(value)
    # Normalise if the LLM returned a 0-10 scale instead of 0-1
    if v > 1.0:
        v = v / 10.0
    return max(lo, min(hi, v))


def _heuristic_validate(
    answer: str,
    chunks: list,
    grounding_threshold: float,
    offensive_threshold: float,
) -> ValidationResult:
    """Word-overlap grounding check used when the LLM call fails."""
    _STOPWORDS = frozenset({
        "that", "this", "with", "from", "have", "been", "will", "your",
        "they", "their", "also", "more", "such", "about", "when", "where",
        "which", "there", "these", "those", "into", "over", "under",
    })

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip())
        if len(s.strip()) > 10
    ]
    evidence_text = " ".join(c.text.lower() for c in chunks)

    supported = 0
    flagged: list = []
    for sent in sentences:
        words = [
            w.lower().strip(".,;:\"'()")
            for w in sent.split()
            if len(w) > 3 and w.lower() not in _STOPWORDS
        ]
        if not words:
            supported += 1
            continue
        threshold = max(1, round(len(words) * 0.3))
        if sum(1 for w in words if w in evidence_text) >= threshold:
            supported += 1
        else:
            flagged.append(sent)

    total = len(sentences) or 1
    grounding = supported / total
    is_valid = grounding >= grounding_threshold

    return ValidationResult(
        is_valid=is_valid,
        grounding_score=round(grounding, 3),
        quality_score=grounding,
        offensive_score=0.0,
        flagged_sentences=flagged,
        validation_note="[heuristic fallback] grounding=" + str(round(grounding, 2)),
    )


def validate(
    query: str,
    answer: str,
    chunks: list,
    grounding_threshold: float = 0.5,
    offensive_threshold: float = 0.5,
) -> ValidationResult:
    """
    Validate the generated answer using the LLM.

    Checks grounding (evidence support), quality (relevance), and
    offensive content. Falls back to heuristic if the LLM call fails.
    """
    try:
        evidence_block = _build_evidence_block(chunks)
        prompt = (
            _VALIDATOR_SYSTEM
            + "\n\n"
            + _VALIDATOR_PROMPT.format(
                query=query,
                answer=answer,
                evidence_block=evidence_block,
            )
        )

        response = get_gemini_model().generate_content(prompt)
        data = _parse_json(response.text.strip())

        grounding = _clamp(data.get("grounding_score", 0.5))
        quality   = _clamp(data.get("quality_score",   0.5))
        offensive = _clamp(data.get("offensive_score", 0.0))
        flagged   = [str(s) for s in data.get("flagged_sentences", []) if s]
        note      = str(data.get("note", "")).strip()

        is_valid = (grounding >= grounding_threshold) and (offensive < offensive_threshold)

        logger.info(
            "validate: grounding=%.2f  quality=%.2f  offensive=%.2f  valid=%s",
            grounding, quality, offensive, is_valid,
        )

        return ValidationResult(
            is_valid=is_valid,
            grounding_score=grounding,
            quality_score=quality,
            offensive_score=offensive,
            flagged_sentences=flagged,
            validation_note=note,
        )

    except Exception as exc:
        logger.warning("validator LLM call failed (%s) -- using heuristic fallback.", exc)
        return _heuristic_validate(answer, chunks, grounding_threshold, offensive_threshold)


def format_answer(answer: str, validation: ValidationResult) -> str:
    """
    Return the answer unchanged if valid.
    Append a plain-text disclaimer if grounding or offensive checks failed.
    """
    if validation.is_valid:
        return answer

    reasons: list = []
    if validation.grounding_score < 0.5:
        reasons.append(
            "grounding score is low (" + str(round(validation.grounding_score, 2)) + ")"
        )
    if validation.offensive_score >= 0.5:
        reasons.append(
            "potential offensive content detected (score "
            + str(round(validation.offensive_score, 2)) + ")"
        )

    disclaimer = "\n\nNote: This response could not be fully verified."
    if reasons:
        disclaimer += " Reason: " + "; ".join(reasons) + "."
    if validation.flagged_sentences:
        disclaimer += (
            "\nFlagged: "
            + " | ".join(s[:120] for s in validation.flagged_sentences[:3])
        )

    return answer + disclaimer
