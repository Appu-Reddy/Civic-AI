import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_CLAIM_SIGNALS: list[re.Pattern] = [
    # Numeric quantities: any number with or without units
    re.compile(r"\b\d[\d,\.]*\s*(?:%|percent|per cent|kg|km|mg|ml|m²|ha|MW|GW|kWh|USD|EUR|GBP|INR)?\b"),
    # Explicit dates and years
    re.compile(
        r"\b(\d{1,2}\s*(st|nd|rd|th)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"|\b(19|20)\d{2}\b)",
        re.IGNORECASE,
    ),
    # Temporal or conditional language (claims about when/if something applies)
    re.compile(r"\b(since|until|by|before|after|during|within|as of|no later than)\b", re.IGNORECASE),
    # Causal or outcome claims
    re.compile(r"\b(results in|leads to|causes|therefore|thus|hence|consequently|as a result)\b", re.IGNORECASE),
    # Definitional or attributive claims
    re.compile(r"\b(is defined as|refers to|means|is known as|is classified as|is described as)\b", re.IGNORECASE),
    # Named entities followed by a factual assertion (proper noun + verb)
    re.compile(r"\b[A-Z][A-Za-z]{2,}\s+(?:was|is|are|were|has|have|had|will|can|must|should)\b"),
    # Superlatives and comparatives that assert a factual ranking
    re.compile(r"\b(largest|smallest|highest|lowest|first|last|most|least|best|worst|only|unique|major|primary|key)\b", re.IGNORECASE),
    # URLs or document/report references
    re.compile(r"\b(report|study|survey|publication|article|chapter|section|figure|table|annex|appendix)\b", re.IGNORECASE),
]


@dataclass
class SentenceCheck:
    sentence: str
    has_claim: bool
    is_supported: bool
    support_source: Optional[str] = None
    flag: str = "ok"


class ValidationResult(BaseModel):
    is_valid: bool
    grounding_score: float
    total_sentences: int        = 0
    claim_sentences: int        = 0
    supported_claims: int       = 0
    unsupported_claims: int     = 0
    flagged_sentences: list[str] = Field(default_factory=list)
    sentence_checks: list[dict]  = Field(default_factory=list)
    validation_note: str        = ""

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid, "grounding_score": round(self.grounding_score, 3),
            "total_sentences": self.total_sentences, "claim_sentences": self.claim_sentences,
            "supported_claims": self.supported_claims, "unsupported_claims": self.unsupported_claims,
            "flagged_sentences": self.flagged_sentences, "validation_note": self.validation_note,
        }


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 10]


def _has_claim_signal(sentence: str) -> bool:
    return any(p.search(sentence) for p in _CLAIM_SIGNALS)


def _find_support(sentence: str, evidence_list: list) -> Optional[str]:
    _STOPWORDS = {
        "that", "this", "with", "from", "have", "been", "will", "your",
        "they", "their", "also", "more", "such", "about", "when", "where",
        "which", "there", "these", "those", "into", "over", "under",
    }
    words = [w.lower().strip(".,;:\"'()") for w in sentence.split() if len(w) > 3 and w.lower() not in _STOPWORDS]
    if not words:
        return None
    threshold = max(1, round(len(words) * 0.3))
    for ev in evidence_list:
        if sum(1 for w in words if w in ev.text.lower()) >= threshold:
            return f"{ev.document_name} p{ev.page_number}"
    return None


def validate_response(generated_response, evidence_list: list, grounding_threshold: float = 0.6) -> ValidationResult:
    sentences = _split_sentences(generated_response.answer)
    checks: list[SentenceCheck] = []
    claim_count = 0
    supported_count = 0
    flagged: list[str] = []

    for sentence in sentences:
        if not _has_claim_signal(sentence):
            checks.append(SentenceCheck(sentence=sentence, has_claim=False, is_supported=True, flag="no_claim"))
            continue

        claim_count += 1

        if re.search(r"\b(not enough|insufficient|not found|no information|unable to find|cannot confirm)\b", sentence, re.IGNORECASE):
            checks.append(SentenceCheck(sentence=sentence, has_claim=True, is_supported=True, flag="ok", support_source="self-declared insufficient"))
            supported_count += 1
            continue

        support_source = _find_support(sentence, evidence_list)
        is_supported = support_source is not None
        if is_supported:
            supported_count += 1
        else:
            flagged.append(sentence)

        checks.append(SentenceCheck(sentence=sentence, has_claim=True, is_supported=is_supported, support_source=support_source, flag="ok" if is_supported else "unsupported"))

    grounding_score = (supported_count / claim_count) if claim_count > 0 else 1.0
    is_valid = grounding_score >= grounding_threshold
    unsupported_count = claim_count - supported_count

    note = (
        f"Response passed. {supported_count}/{claim_count} claims grounded (score={grounding_score:.2f})."
        if is_valid else
        f"Response has {unsupported_count} unsupported claim(s) (score={grounding_score:.2f}, threshold={grounding_threshold})."
    )
    return ValidationResult(
        is_valid=is_valid, grounding_score=grounding_score,
        total_sentences=len(sentences), claim_sentences=claim_count,
        supported_claims=supported_count, unsupported_claims=unsupported_count,
        flagged_sentences=flagged,
        sentence_checks=[{"sentence": c.sentence[:100], "has_claim": c.has_claim, "is_supported": c.is_supported, "support_source": c.support_source, "flag": c.flag} for c in checks],
        validation_note=note,
    )


def format_validated_response(generated_response, validation_result: ValidationResult) -> str:
    if validation_result.is_valid:
        return generated_response.answer
    disclaimer = (
        "\n\n⚠️ Note: Some parts of this response could not be fully verified "
        "against available documents. Please verify the following with official sources before acting:\n"
        + "\n".join(f"  • {s[:120]}" for s in validation_result.flagged_sentences[:3])
    )
    return generated_response.answer + disclaimer