"""
StepDefiner agent

Single responsibility: turn one abstract plan step (a "goal") into a
concrete, self-contained, retrieval-ready subquery - resolving any
references to prior steps' answers along the way (e.g. pronouns like
"that language" or "the player"). This agent does NOT retrieve, does NOT
plan, and does NOT call any other agent - orchestration/looping across
steps is graph.py's job (Phase-4).
"""

from typing import Dict, List

from llm_client import call_llm, parse_json_response, LLMCallError

SYSTEM_PROMPT = """You are the StepDefiner agent in a multi-agent RAG system (MA-RAG).

	Your job: given the user's original query, the current plan step's abstract
	goal, and the history of previously completed steps (each with its own
	subquery and answer), produce ONE concrete, self-contained subquery that a
	retriever can search for directly.

	Critical rule: if history is non-empty and the current step's goal
	references something resolved by a prior step (a pronoun like "that
	language", "the player", "it", or any implicit dependency on a prior
	answer), you MUST rewrite the subquery to include the actual resolved
	value from the prior answer. Never repeat the abstract goal verbatim if it
	still contains an unresolved reference - the retriever has no memory of
	prior steps and only sees the subquery you produce.

	Output ONLY the final JSON object - no prose, no markdown code fences.

	Output JSON schema:
	{
	"step_id": <int, matching current_step's step_id>,
	"subquery": "<concrete, self-contained subquery text>"
	}

	### Example (with history - reference must be resolved)
	Original query: "How does exception handling in the reference codebase's language compare to Python's?"
	History:
	Step 1: goal="Identify the primary language used in the reference codebase", subquery="What is the primary language used in the reference codebase?", answer="Java"
	Current step: {"step_id": 2, "goal": "how does that language handle exceptions"}
	Reasoning: "that language" refers to Step 1's answer, "Java". The subquery
	must name "Java" explicitly - repeating "that language" verbatim would give
	the retriever nothing concrete to search for.
	Output:
	{"step_id": 2, "subquery": "How does Java handle exceptions?"}

	### Example (no history - first step, nothing to resolve)
	Original query: "What is a Python decorator?"
	History: (empty)
	Current step: {"step_id": 1, "goal": "What is a Python decorator?"}
	Reasoning: No prior context exists to resolve, so the subquery is just the
	goal as-is.
	Output:
	{"step_id": 1, "subquery": "What is a Python decorator?"}
"""


def _format_history(history: List[Dict]) -> str:
	if not history:
		return "(empty - this is the first step, nothing to resolve)"
	lines = []
	for entry in history:
		lines.append(
			f"  Step {entry.get('step_id')}: goal={entry.get('goal', '')!r}, "
			f"subquery={entry.get('subquery', '')!r}, answer={entry.get('answer', '')!r}"
		)
	return "\n".join(lines)


def define_step(original_query: str, current_step: Dict, history: List[Dict]) -> Dict:
	"""
	Produce a concrete, retrieval-ready subquery for `current_step`, grounded
	in `history` (prior steps' subqueries + answers, resolving any
	references the step's abstract goal makes to them).
	"""
	user_prompt = (
		f"Original query: {original_query!r}\n\n"
		f"History:\n{_format_history(history)}\n\n"
		f"Current step: {current_step!r}\n\n"
		"Produce the JSON subquery now."
	)

	raw_response = call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
	result = parse_json_response(raw_response)

	required_keys = {"step_id", "subquery"}
	missing = required_keys - result.keys()
	if missing:
		raise LLMCallError(
			f"StepDefiner response missing keys {missing}. Raw output:\n{raw_response}"
		)
	if not str(result.get("subquery", "")).strip():
		raise LLMCallError(f"StepDefiner returned an empty subquery. Raw output:\n{raw_response}")

	return result


if __name__ == "__main__":
	original_query = "How does exception handling in the reference codebase's language compare to Python's?"
	history = [
		{
			"step_id": 1,
			"goal": "Identify the primary language used in the reference codebase",
			"subquery": "What is the primary language used in the reference codebase?",
			"answer": "Java",
		},
	]
	current_step = {"step_id": 2, "goal": "how does that language handle exceptions"}

	result = define_step(original_query, current_step, history)
	print(result)