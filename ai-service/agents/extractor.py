"""
Extractor agent

Single responsibility: given retrieved chunks and the current subquery,
filter/condense them down to only the directly-relevant evidence (per the
MA-RAG paper: reduce noise, avoid lost-in-the-middle, keep context concise
for the QA agent). Does NOT retrieve, does NOT answer, does NOT call any
other agent.
"""

from typing import Dict, List

from llm_client import call_llm, parse_json_response, LLMCallError
from config import NO_EVIDENCE_MARKER

SYSTEM_PROMPT = f"""You are the Extractor agent in a multi-agent RAG system (MA-RAG).

	You will be given a subquery and a set of retrieved text chunks, each
	labeled with its chunk_id. Your job:

	1. Discard chunks that are irrelevant to the subquery entirely.
	2. From the remaining chunks, extract ONLY the specific spans/sentences
	that directly help answer the subquery - do not include the full
	chunk text, do not include tangential information, do not repeat
	filler. Condense aggressively.
	3. Evidence requirements:
		- Maximum 80 words.
		- Maximum 3 sentences.
		- Never copy entire paragraphs.
		- Keep evidence concise.
		- Return valid JSON only.
	4. If NONE of the retrieved chunks are relevant to the subquery, set
	"evidence" to exactly this string: "{NO_EVIDENCE_MARKER}" - do not
	fabricate evidence, do not leave it blank, do not explain why nothing
	was found inside the evidence field itself.

	Output ONLY the final JSON object - no prose, no markdown code fences.

	Output JSON schema:
	{{
	"subquery": "<the subquery, verbatim>",
	"evidence": "<condensed, relevant text only, with inline chunk_id tags>",
	"source_chunk_ids": ["<chunk_id>", ...]
	}}

	If evidence is "{NO_EVIDENCE_MARKER}", source_chunk_ids must be an empty list.

	### Example (relevant evidence found)
	Subquery: "How does Python implement context managers?"
	Chunks:
	[chunk_id: c001] "Python installation on Windows requires downloading the installer from python.org..."
	[chunk_id: c002] "Context managers in Python are implemented via the __enter__ and __exit__ dunder methods, and can also be created using the @contextmanager decorator from the contextlib module..."
	Reasoning: c001 is about installation, irrelevant to context managers -
	discard entirely. c002 directly answers the subquery - extract it.
	Output:
	{{
	"subquery": "How does Python implement context managers?",
	"evidence": "Context managers are implemented via the __enter__ and __exit__ dunder methods, or via the @contextmanager decorator from contextlib. [chunk_id: c002]",
	"source_chunk_ids": ["c002"]
	}}

	### Example (no relevant evidence)
	Subquery: "How does Rust's borrow checker work?"
	Chunks:
	[chunk_id: c010] "Java uses a garbage collector to manage memory automatically..."
	[chunk_id: c011] "Python uses reference counting plus a cyclic garbage collector..."
	Reasoning: Neither chunk discusses Rust or a borrow checker at all.
	Output:
	{{
	"subquery": "How does Rust's borrow checker work?",
	"evidence": "{NO_EVIDENCE_MARKER}",
	"source_chunk_ids": []
	}}

	NOTE: Maintain the evidence short in ONE (or) TWO lines ONLY
"""


def _format_chunks(retrieved_chunks: List[Dict]) -> str:
	if not retrieved_chunks:
		return "(no chunks were retrieved)"
	lines = []
	for chunk in retrieved_chunks:
		lines.append(f"[chunk_id: {chunk['chunk_id']}] {chunk['text']}")
	return "\n\n".join(lines)


def extract(subquery: str, retrieved_chunks: List[Dict]) -> Dict:
	user_prompt = (
		f"Subquery: {subquery!r}\n\n"
		f"Retrieved chunks:\n{_format_chunks(retrieved_chunks)}\n\n"
		"Produce the JSON extraction now."
	)

	raw_response = call_llm(
		system_prompt=SYSTEM_PROMPT, 
		user_prompt=user_prompt
	)
	result = parse_json_response(raw_response)

	required_keys = {"subquery", "evidence", "source_chunk_ids"}
	missing = required_keys - result.keys()
	if missing:
		raise LLMCallError(
			f"Extractor response missing keys {missing}. Raw output:\n{raw_response}"
		)

	# Defensive normalization: an LLM that says "no evidence" but still forgot
	# to empty source_chunk_ids shouldn't propagate a false traceability trail.
	if result["evidence"].strip() == NO_EVIDENCE_MARKER:
		result["source_chunk_ids"] = []
	elif not result["evidence"].strip():
		raise LLMCallError(
			f"Extractor returned an empty evidence string instead of the "
			f"'{NO_EVIDENCE_MARKER}' marker. Raw output:\n{raw_response}"
		)

	return result


# TESTING

# if __name__ == "__main__":
# 	from retriever import retrieve

# 	test_subquery = "Explain python variables."
# 	chunks = retrieve(test_subquery, top_k=5)

# 	result = extract(test_subquery, chunks)
# 	print("Evidence:")
# 	print(result["evidence"])
# 	print("\nSource chunk IDs:", result["source_chunk_ids"])