"""
QA agent

Single responsibility: synthesize a step-level answer from the subquery and
the extracted evidence (extractor.py's output). The answer MUST be grounded
only in the given evidence - no outside knowledge, no guessing. Does NOT
retrieve, does NOT extract, does NOT call any other agent.
"""

from typing import Dict

from llm_client import call_llm, parse_json_response, LLMCallError
from config import NO_EVIDENCE_MARKER
from log import stage

CANNOT_ANSWER_TEXT = "Cannot answer from available evidence."

SYSTEM_PROMPT = """You are the QA agent in a multi-agent RAG system (MA-RAG).

	You will be given a subquery and a block of evidence that was already
	retrieved and condensed for you. Your job: answer the subquery using ONLY
	the given evidence. Do NOT use any outside knowledge, do NOT fill gaps with
	assumptions, and do NOT guess beyond what the evidence actually supports.

	If the evidence genuinely supports an answer, give a clear, direct answer
	and set "grounded" to true.

	If the evidence does not contain enough information to answer the subquery
	(including if it is empty, irrelevant, or explicitly says no evidence was
	found), do NOT hallucinate an answer. Instead say plainly that it cannot be
	answered from the available evidence, and set "grounded" to false.

	Output ONLY the final JSON object - no prose, no markdown code fences.

	Output JSON schema:
	{
	"subquery": "<the subquery, verbatim>",
	"answer": "<the answer, or an honest 'cannot answer' statement>",
	"grounded": true|false
	}

	### Example (evidence supports an answer)
	Subquery: "How does Python implement context managers?"
	Evidence: "Context managers are implemented via the __enter__ and __exit__ dunder methods, or via the @contextmanager decorator from contextlib. [chunk_id: c002]"
	Output:
	{
	"subquery": "How does Python implement context managers?",
	"answer": "Python implements context managers through the __enter__ and __exit__ dunder methods, or more simply via the @contextmanager decorator from the contextlib module.",
	"grounded": true
	}

	### Example (evidence does not support an answer)
	Subquery: "How does Rust's borrow checker work?"
	Evidence: "No relevant evidence found"
	Output:
	{
	"subquery": "How does Rust's borrow checker work?",
	"answer": "Cannot answer from available evidence.",
	"grounded": false
	}

	NOTE: 
	1. Maintain the answer small but closely perfect.
	2. Do not use Markdown formatting.
	3. Do not use **bold**, bullet symbols, or code blocks.
	4. Return plain text only.
"""


def answer(subquery: str, evidence: str) -> Dict:
	"""
	Synthesize a step-level answer to `subquery` grounded strictly in
	`evidence`. If evidence is extractor.py's explicit "no evidence" marker,
	short-circuit without an LLM call at all - this guarantees an honest,
	deterministic non-answer rather than trusting the model to always
	recognize and honor that case.
	"""
	if evidence.strip() == NO_EVIDENCE_MARKER:
		return {
			"subquery": subquery,
			"answer": CANNOT_ANSWER_TEXT,
			"grounded": False,
		}

	user_prompt = f"Subquery: {subquery!r}\n\nEvidence: {evidence!r}\n\nProduce the JSON answer now."

	raw_response = call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
	result = parse_json_response(raw_response)

	required_keys = {"subquery", "answer", "grounded"}
	missing = required_keys - result.keys()
	if missing:
		raise LLMCallError(
			f"QA response missing keys {missing}. Raw output:\n{raw_response}"
		)
	if not isinstance(result["grounded"], bool):
		raise LLMCallError(
			f"QA response 'grounded' field is not a boolean. Raw output:\n{raw_response}"
		)

	stage("Result Loaded.")

	return result


# TESTING

# if __name__ == "__main__":
# 	answerable_evidence = (
# 		"Context managers are implemented via the __enter__ and __exit__ "
# 		"dunder methods, or via the @contextmanager decorator from contextlib. "
# 		"[chunk_id: c002]"
# 	)
# 	result_ok = answer("How does Python implement context managers?", answerable_evidence)
# 	print("=== Answerable case ===")
# 	print(result_ok)
# 	assert result_ok["grounded"] is True

# 	result_no_evidence = answer(
# 		"How does Rust's borrow checker work?",
# 		NO_EVIDENCE_MARKER,
# 	)
# 	print("\n=== No-evidence case ===")
# 	print(result_no_evidence)
# 	assert result_no_evidence["grounded"] is False