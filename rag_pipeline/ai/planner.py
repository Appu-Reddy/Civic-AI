"""
planner.py — Query planner

Single LLM call that analyses the user query and returns:
    is_multi_hop : bool          — True when the query needs multiple retrieval steps
    steps        : list[dict]    — each dict has a "sub_query" key
                                   (single-hop → exactly one step equal to the original query)
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

MAX_STEPS: int = 3

@dataclass
class QueryPlan:
    raw_query: str
    is_multi_hop: bool
    steps: list[dict] = field(default_factory=list)
    # steps is a list of {"sub_query": "..."}

    def sub_queries(self) -> list[str]:
        return [s["sub_query"] for s in self.steps]


_PLANNER_PROMPT = """\
You are an expert assistant that analyses user questions about PDF documents.

Given the user query below, decide:
1. Does the query require multiple sequential retrieval steps to answer correctly?
   Set "is_multi_hop" to true ONLY when the query explicitly needs information from
   two or more clearly separate aspects that cannot be answered in a single lookup.
   Prefer false for straightforward questions.

2. Decompose the query into AT MOST {max_steps} sequential sub-queries.
   - If is_multi_hop is false, produce exactly ONE step whose sub_query is the original query.
   - If is_multi_hop is true, produce 2-{max_steps} steps where each sub_query is a
     specific, self-contained question. The LAST step must synthesise or compare the
     prior steps (e.g. "Given the above findings, compare / summarise / conclude...").

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "is_multi_hop": <true|false>,
  "steps": [
    {{"sub_query": "<specific question for step 1>"}},
    {{"sub_query": "<specific question for step 2>"}},
    ...
  ]
}}

Rules:
- Maximum {max_steps} steps. Never exceed this.
- Every step must have "sub_query" as a non-empty string.
- Do NOT wrap the JSON in markdown fences.

User query: {query}
"""

def _clean_json(text: str) -> str:
    """
    Robustly extract and clean a JSON object from raw LLM output.

    Order of operations matters:
    1. Normalize escaped formatting markers returned by some LLM responses
    2. Strip markdown fences (```json ... ```)
    3. Extract the outermost {...} block — handles leading/trailing prose
    """
    # 1. Normalize JSON that was returned with literal escaped line breaks.
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ").replace("\\t", " ")

    # 2. Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # 3. Extract the outermost JSON object before touching control chars
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()

    return text


def _parse_json(text: str) -> dict:
    cleaned = _clean_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from response: {e}\n{cleaned[:300]}")

def plan_query(query: str) -> QueryPlan:
    """
    Call the LLM once to produce a QueryPlan.

    Returns a QueryPlan with:
        is_multi_hop  — whether multiple hops are needed
        steps         — list of {"sub_query": "..."} dicts (1–MAX_STEPS items)

    On any LLM or parse failure falls back to a safe single-step plan.
    """
    try:
        model = get_gemini_model()
        response = model.generate_content(
            _PLANNER_PROMPT.format(query=query, max_steps=MAX_STEPS)
        )
        data = _parse_json(response.text.strip())

        is_multi_hop = bool(data.get("is_multi_hop", False))
        raw_steps = data.get("steps", [])

        if not isinstance(raw_steps, list) or len(raw_steps) == 0:
            raise ValueError("'steps' must be a non-empty list")

        steps: list[dict] = []
        for i, step in enumerate(raw_steps[:MAX_STEPS]):
            sub_query = str(step.get("sub_query", "")).strip()
            if not sub_query:
                raise ValueError(f"Step {i + 1} has an empty sub_query")
            steps.append({"sub_query": sub_query})

        # Enforce consistency: single-hop must have exactly one step
        if not is_multi_hop:
            steps = [{"sub_query": query}]
        return QueryPlan(raw_query=query, is_multi_hop=is_multi_hop, steps=steps)

    except Exception as exc:
        logger.warning("plan_query failed (%s) — falling back to single-step plan.", exc)
        return QueryPlan(
            raw_query=query,
            is_multi_hop=False,
            steps=[{"sub_query": query}],
        )
