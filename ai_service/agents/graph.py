from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_AI_SERVICE_AGENTS_DIR = Path(__file__).resolve().parent
if str(_AI_SERVICE_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_SERVICE_AGENTS_DIR))

from config import MAX_STEPS
from extractor import extract
from llm_client import call_llm
from planner import plan
from qa import answer
from retriever import retrieve
from stepDefiner import define_step

LOGGER = logging.getLogger(__name__)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_step(original_query: str, current_step: Dict, history: List[Dict], top_k: int = 5) -> Dict:
    step_id = int(current_step.get("step_id", 0))
    goal = str(current_step.get("goal", ""))

    try:
        step_def = define_step(original_query, current_step, history)
        subquery = str(step_def.get("subquery", "")).strip()

        retrieved_chunks = retrieve(subquery, top_k=top_k)
        retrieved_chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks if chunk.get("chunk_id")]

        extraction = extract(subquery, retrieved_chunks)
        evidence = str(extraction.get("evidence", "")).strip()
        source_chunk_ids = [str(chunk_id) for chunk_id in extraction.get("source_chunk_ids", [])]

        qa_result = answer(subquery, evidence)
        return {
            "step_id": step_id,
            "goal": goal,
            "subquery": subquery,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "source_chunk_ids": source_chunk_ids,
            "evidence": evidence,
            "answer": str(qa_result.get("answer", "")).strip(),
            "grounded": bool(qa_result.get("grounded", False)),
            "status": "ok",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Step %s failed: %s", step_id, exc)
        return {
            "step_id": step_id,
            "goal": goal,
            "subquery": "",
            "retrieved_chunk_ids": [],
            "source_chunk_ids": [],
            "evidence": "",
            "answer": "",
            "grounded": False,
            "status": "error",
            "error": str(exc),
        }


def synthesize_final_answer(original_query: str, completed_steps: List[Dict]) -> str:
    if not completed_steps:
        return "I could not produce a final answer from the available evidence."

    if len(completed_steps) == 1:
        step = completed_steps[0]
        if step.get("status") == "error":
            return "I could not produce a fully grounded answer because the step failed."
        if not step.get("grounded", False):
            return "I could not produce a fully grounded answer because the available evidence was insufficient."
        return str(step.get("answer", "")).strip()

    has_gap = any(step.get("status") != "ok" or not step.get("grounded", False) for step in completed_steps)
    step_summaries = "\n".join(
        f"- Step {step.get('step_id')}: goal={step.get('goal')!r}; answer={step.get('answer')!r}"
        for step in completed_steps
    )

    synthesis_prompt = (
        f"You are synthesizing the final answer for a multi-step RAG workflow.\n"
        f"Original query: {original_query}\n\n"
        f"Completed step summaries:\n{step_summaries}\n\n"
        "Produce one direct, coherent final answer that combines the steps.\n"
        "Do not simply concatenate the step answers.\n\n"
        "Return ONLY the final answer as plain text.\n"
        "Do NOT return JSON.\n"
        "Do NOT use markdown.\n"
    )
    if has_gap:
        synthesis_prompt += (
            " Some steps were incomplete, ungrounded, or errored. "
            "Acknowledge that limitation explicitly and avoid presenting a confident answer that overstates the evidence."
        )

    try:
        return call_llm(
            system_prompt=(
                "You synthesize a final answer for a multi-agent RAG workflow. "
                "Always return plain text only. "
                "Never return JSON. "
                "Never use markdown."
            ),
            user_prompt=synthesis_prompt,
            temperature=0.0,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Final synthesis failed: %s", exc)
        return "I could not synthesize a final answer because the synthesis step failed."


def run_marag(query: str, top_k: int = 5, max_steps: int = MAX_STEPS) -> Dict:
    try:
        plan_result = plan(query)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Planning failed for query %r: %s", query, exc)
        return {
            "query": query,
            "final_answer": f"Planning failed: {exc}",
            "steps": [],
            "num_steps_planned": 0,
            "num_steps_executed": 0,
            "is_multi_step": False,
        }

    original_steps = list(plan_result.get("steps", []))
    num_steps_planned = len(original_steps)
    steps_to_run = original_steps[:max_steps]

    if len(original_steps) > max_steps:
        LOGGER.warning("Planner returned %d steps; truncating to %d", len(original_steps), max_steps)

    history: List[Dict] = []
    executed_steps: List[Dict] = []

    for _,current_step in enumerate(steps_to_run, start=1):
        step_result = run_step(query, current_step, history, top_k=top_k)
        executed_steps.append(step_result)
        history.append(step_result)

    final_answer = synthesize_final_answer(query, executed_steps)

    return {
        "query": query,
        "final_answer": final_answer,
        "steps": executed_steps,
        "num_steps_planned": num_steps_planned,
        "num_steps_executed": len(executed_steps),
        "is_multi_step": bool(plan_result.get("is_multi_step", False)),
    }


# TESTING

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

#     queries = [
#         "What is a Python tuple?",
#         "How different are Collections in Java and Python?",
#     ]

#     for query in queries:
#         result = run_marag(query)
#         print(f"\n=== Query: {query} ===")
#         print("Final answer:")
#         print(result["final_answer"])
#         print("\nTrace:")
#         for step in result["steps"]:
#             print(
#                 f"- goal={step.get('goal')!r} | subquery={step.get('subquery')!r} | "
#                 f"answer={step.get('answer')!r} | grounded={step.get('grounded')}"
#             )
