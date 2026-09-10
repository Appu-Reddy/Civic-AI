import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_CATEGORY_SIGNALS: dict[str, list[str]] = {
    "definition": [
        "is defined as", "refers to", "means", "is known as", "is called",
        "denotes", "represents", "in the context of", "is described as",
        "can be understood as", "is a type of", "is an example of",
    ],
    "quantitative": [
        "percent", "per cent", "%", "million", "billion", "thousand",
        "total", "average", "rate", "ratio", "number of", "amount",
        "increased by", "decreased by", "grew", "declined", "estimated",
    ],
    "process": [
        "step", "process", "procedure", "method", "approach", "mechanism",
        "in order to", "how to", "implemented", "carried out", "conducted",
        "first", "then", "next", "finally", "followed by",
    ],
    "causal": [
        "because", "therefore", "thus", "hence", "as a result",
        "leads to", "causes", "results in", "consequently", "due to",
        "attributed to", "enables", "supports", "contributes to",
    ],
    "comparative": [
        "compared to", "in contrast", "whereas", "however", "on the other hand",
        "unlike", "similar to", "difference between", "better than", "worse than",
        "higher than", "lower than", "more than", "less than",
    ],
}


def _categorise_chunk(text: str) -> str:
    text_lower = text.lower()
    scores = {cat: sum(1 for s in signals if s in text_lower) for cat, signals in _CATEGORY_SIGNALS.items()}
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "general"


def _relevance_score(chunk_text: str, objective: str, step_keywords: list[str], domain: str, step_domain: Optional[str]) -> float:
    text_lower = chunk_text.lower()
    score = 0.0
    if step_keywords:
        score += 0.5 * (sum(1 for kw in step_keywords if kw.lower() in text_lower) / len(step_keywords))
    obj_words = [w for w in re.split(r"\W+", objective.lower()) if len(w) > 3]
    if obj_words:
        score += 0.3 * (sum(1 for w in obj_words if w in text_lower) / len(obj_words))
    if step_domain and domain and step_domain.lower() == domain.lower():
        score += 0.2
    return min(score, 1.0)


def extract_evidence(
    retrieval_results: list,
    step_objective: str,
    step_keywords: list[str],
    step_domain: Optional[str] = None,
    min_score: float = 0.1,
    max_evidence: int = 6,
) -> list:
    from memory.history import EvidenceRef

    if not retrieval_results:
        return []

    scored: list[tuple[float, object]] = []
    for result in retrieval_results:
        if not result.text:
            continue
        score = _relevance_score(result.text, step_objective, step_keywords, result.domain, step_domain)
        if score >= min_score:
            scored.append((score, result))

    scored.sort(key=lambda x: x[0], reverse=True)

    evidence_list = []
    for score, result in scored[:max_evidence]:
        ev = EvidenceRef(
            chunk_id=result.chunk_id, document_name=result.document_name,
            page_number=result.page_number, section=result.section,
            text=result.text, source=result.source,
        )
        ev.__dict__["category"] = _categorise_chunk(result.text)
        ev.__dict__["relevance_score"] = round(score, 4)
        evidence_list.append(ev)

    return evidence_list


def evidence_to_context_string(evidence_list: list) -> str:
    if not evidence_list:
        return "No relevant evidence found."
    lines = ["Retrieved Evidence:"]
    for i, ev in enumerate(evidence_list, start=1):
        category = ev.__dict__.get("category", "general")
        section_str = f" | {ev.section}" if ev.section else ""
        lines.append(f"\n[{i}] {ev.document_name} â€” p{ev.page_number}{section_str} ({category})")
        lines.append(f"    {ev.text.strip()}")
    return "\n".join(lines)


## TESTING ##
# if __name__ == "__main__":
#     import sys
#
#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))
#
#     from retrieval.hybrid import RetrievalResult
#
#     mock_results = [
#         RetrievalResult(chunk_id="c1", rank=1, rrf_score=0.9, document_name="ClimateReport2023", page_number=5,
#             section="Findings", domain="climate_report2023", source="ClimateReport2023.pdf",
#             text="Global temperatures have risen by 1.1 degrees Celsius since pre-industrial levels.",
#             faiss_rank=1, graph_rank=None, retrieval_sources=["faiss"]),
#         RetrievalResult(chunk_id="c2", rank=2, rrf_score=0.8, document_name="ClimateReport2023", page_number=8,
#             section="Impact", domain="climate_report2023", source="ClimateReport2023.pdf",
#             text="Sea levels are projected to rise by 0.3 to 1.0 metres by 2100 under high-emission scenarios.",
#             faiss_rank=2, graph_rank=None, retrieval_sources=["faiss"]),
#         RetrievalResult(chunk_id="c3", rank=3, rrf_score=0.6, document_name="ClimateReport2023", page_number=12,
#             section="Methods", domain="climate_report2023", source="ClimateReport2023.pdf",
#             text="The study used satellite data combined with ground station measurements to compute global averages.",
#             faiss_rank=3, graph_rank=None, retrieval_sources=["faiss"]),
#     ]
#
#     print(f"\n{'='*60}")
#     print("EXTRACTOR — generic test\n")
#
#     ev1 = extract_evidence(mock_results, "Retrieve key findings about temperature rise",
#                            ["temperature", "findings", "rise"], step_domain="climate_report2023")
#     for ev in ev1:
#         print(f"  [{ev.__dict__['category']:12}] {ev.__dict__['relevance_score']}  {ev.document_name} p{ev.page_number}: {ev.text[:80]}...")
#
#     print(f"\n{evidence_to_context_string(ev1)}")
#     print("\nExtractor test complete.")
