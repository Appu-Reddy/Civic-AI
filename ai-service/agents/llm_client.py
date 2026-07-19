import json
import logging
import os
import time
from typing import Any, Optional

import google.generativeai as genai

from config import DEFAULT_MODEL # type: ignore

LOGGER = logging.getLogger(__name__)

MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0

_configured = False

class LLMCallError(Exception):
	"""Raised when call_llm() fails after all retries are exhausted, or when
	an agent's response otherwise can't be trusted (missing/empty fields)."""


class LLMJSONParseError(LLMCallError):
	"""Raised when the LLM's response could not be parsed as JSON."""


def _ensure_configured() -> None:
	global _configured
	if _configured:
		return

	api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
	if not api_key:
		raise LLMCallError(
			"No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
		)
	genai.configure(api_key=api_key)
	_configured = True


def call_llm(
	system_prompt: str,
	user_prompt: str,
	model: str = DEFAULT_MODEL,
	max_tokens: int = 1024,
	temperature: float = 0.0,
) -> str:
	_ensure_configured()

	generative_model = genai.GenerativeModel(
		model_name=model,
		system_instruction=system_prompt,
	)
	generation_config = genai.types.GenerationConfig(
		temperature=temperature,
		max_output_tokens=max_tokens,
	)

	last_error: Optional[Exception] = None
	backoff = INITIAL_BACKOFF_SECONDS

	for attempt in range(MAX_RETRIES + 1):
		try:
			response = generative_model.generate_content(
				user_prompt,
				generation_config=generation_config,
			)

			text = (response.text or "").strip()
			if not text:
				raise LLMCallError(f"LLM returned an empty response. Full response object: {response}")

			return text

		except LLMCallError:
			# Not transient (empty response) - don't retry, fail immediately.
			raise
		except Exception as exc:  # network errors, rate limits, 5xx, etc.
			last_error = exc
			if attempt < MAX_RETRIES:
				LOGGER.warning(
					"call_llm attempt %d/%d failed: %s. Retrying in %.1fs...",
					attempt + 1, MAX_RETRIES + 1, exc, backoff,
				)
				time.sleep(backoff)
				backoff *= BACKOFF_MULTIPLIER
			else:
				LOGGER.error("call_llm failed after %d attempts: %s", MAX_RETRIES + 1, exc)

	raise LLMCallError(
		f"LLM call failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}"
	) from last_error


def parse_json_response(raw_response: str) -> Any:
	cleaned = raw_response.strip()

	if cleaned.startswith("```"):
		cleaned = cleaned.strip("`")
		stripped_lower = cleaned.lstrip().lower()
		if stripped_lower.startswith("json"):
			# remove the leading "json" language tag left after stripping backticks
			cleaned = cleaned.lstrip()[4:]
		cleaned = cleaned.strip()

	try:
		return json.loads(cleaned)
	except json.JSONDecodeError as exc:
		raise LLMJSONParseError(
			f"Failed to parse LLM output as JSON: {exc}. Raw LLM output was:\n{raw_response}"
		) from exc


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

	# Plain-text smoke test
	text_result = call_llm(
		system_prompt="You are a helpful assistant. Respond in one short sentence.",
		user_prompt="What is the capital of France?",
	)
	print("Plain text response:", text_result)

	# JSON smoke test
	json_result = call_llm(
		system_prompt='Respond ONLY with JSON: {"answer": "<one word>"}. No prose, no markdown fences.',
		user_prompt="What is the capital of France?",
	)
	print("Raw JSON text:", json_result)
	print("Parsed:", parse_json_response(json_result))