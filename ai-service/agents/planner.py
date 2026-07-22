"""
Planner agent

Single responsibility: given the user's original query, decide whether it
requires multi-step decomposition (multi-hop) or can be answered directly
in one step (single-hop), and if multi-hop, produce an ordered list of
sub-goals. This agent does NOT retrieve, does NOT define concrete
subqueries, and does NOT call any other agent - looping between steps is
graph.py's job (Phase-4).
"""

from typing import Dict
from log import stage

from llm_client import call_llm, parse_json_response, LLMCallError

SYSTEM_PROMPT = """You are the Planner agent in a multi-agent RAG system (MA-RAG).

	Your job: given a user's question, decide if it requires multiple sequential
	reasoning steps to answer (multi-hop), or if it can be answered directly in
	one step (single-hop). If multi-hop, decompose it into an ordered list of
	steps, where a later step's goal may depend on an earlier step's answer.

	Think step by step about what information is needed and in what order (see
	the reasoning shown in each example below), then output ONLY the final JSON
	object - no prose, no markdown code fences, nothing else before or after it.

	Output JSON schema:
	{
	"original_query": "<the original query, verbatim>",
	"is_multi_step": true|false,
	"steps": [
		{"step_id": 1, "goal": "<goal of this step>"},
		...
	]
	}

	If is_multi_step is false, "steps" must contain exactly ONE entry whose
	"goal" equals the original query, verbatim.

	### Example 1 (multi-hop)
	Query: "Which team did the player born in 1990 who won the Golden Boot in 2018 play for at the time?"
	Reasoning: To answer this we first need to identify WHO the player is
	(born 1990, won the 2018 Golden Boot), then find WHICH TEAM they played for
	at that time. These are two dependent lookups, so this needs decomposition.
	Output:
	{
	"original_query": "Which team did the player born in 1990 who won the Golden Boot in 2018 play for at the time?",
	"is_multi_step": true,
	"steps": [
		{"step_id": 1, "goal": "Identify the player born in 1990 who won the Golden Boot in 2018"},
		{"step_id": 2, "goal": "Find which team that player played for at the time"}
	]
	}

	### Example 2 (multi-hop)
	Query: "How does exception handling in Java compare to how Python does it, and which approach does C++ follow?"
	Reasoning: This requires understanding Java's exception model, then Python's,
	then comparing the two, then separately checking which of those two models
	C++'s exception handling resembles. Multiple distinct lookups feed one final
	comparison, so this needs decomposition.
	Output:
	{
	"original_query": "How does exception handling in Java compare to how Python does it, and which approach does C++ follow?",
	"is_multi_step": true,
	"steps": [
		{"step_id": 1, "goal": "Describe how Java implements exception handling"},
		{"step_id": 2, "goal": "Describe how Python implements exception handling"},
		{"step_id": 3, "goal": "Compare Java's and Python's exception handling approaches"},
		{"step_id": 4, "goal": "Determine which of these approaches (Java-style or Python-style) C++'s exception handling follows"}
	]
	}

	### Example 3 (single-hop, no decomposition needed)
	Query: "What is a Python decorator?"
	Reasoning: This is a single, self-contained factual question. It does not
	depend on any intermediate lookup or answer - it can be answered directly
	in one retrieval + answer step, so no decomposition is needed.
	Output:
	{
	"original_query": "What is a Python decorator?",
	"is_multi_step": false,
	"steps": [
		{"step_id": 1, "goal": "What is a Python decorator?"}
	]
	}

	NOTE: Try to maintain minimum number of steps with a maximum of 3 or 4 steps ONLY.

"""


def plan(query: str) -> Dict:
	
	user_prompt = f"Query: {query!r}\n\nProduce the JSON plan now."

	raw_response = call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
	result = parse_json_response(raw_response)

	required_keys = {"original_query", "is_multi_step", "steps"}
	missing = required_keys - result.keys()
	if missing:
		raise LLMCallError(
			f"Planner response missing keys {missing}. Raw output:\n{raw_response}"
		)
	
	if not result["steps"]:
		raise LLMCallError(f"Planner returned an empty steps list. Raw output:\n{raw_response}")
	
	if not result["is_multi_step"] and len(result["steps"]) != 1:
		raise LLMCallError(
			f"Planner said is_multi_step=False but returned {len(result['steps'])} steps "
			f"(expected exactly 1). Raw output:\n{raw_response}"
		)

	stage("Planning completed.")

	return result


if __name__ == "__main__":
	single_hop_query = "What is a Python decorator?"
	multi_hop_query = (
		"How does exception handling in Java compare to how Python does it, "
		"and which approach does C++ follow?"
	)

	print("=== Single-hop plan ===")
	print(plan(single_hop_query))

	print("\n=== Multi-hop plan ===")
	print(plan(multi_hop_query))