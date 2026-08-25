import json
import os
import threading
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv

from config import DEFAULT_MODEL  # type: ignore

load_dotenv()

class LLMCallError(Exception):
	"""Raised when call_llm() fails."""


class LLMJSONParseError(LLMCallError):
	"""Raised when JSON parsing fails."""

_API_KEYS = [
	os.getenv("GEMINI_API_KEY_1"),
	os.getenv("GEMINI_API_KEY_2"),
	os.getenv("GEMINI_API_KEY_3"),
]

_API_KEYS = [key for key in _API_KEYS if key]

if not _API_KEYS:
	raise LLMCallError(
		"No Gemini API keys found. "
		"Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3 in .env"
	)

_current_key_index = 0
_key_lock = threading.Lock()

def _get_next_api_key() -> str:
	"""
	Round-robin API key selection.
	
	Example:
	KEY_1 -> KEY_2 -> KEY_3 -> KEY_1
	"""
	global _current_key_index
	with _key_lock:
		key = _API_KEYS[_current_key_index]
		_current_key_index = (
			_current_key_index + 1
		) % len(_API_KEYS)

	return key


def call_llm(
	system_prompt: str,
	user_prompt: str,
	model: str = DEFAULT_MODEL,
	temperature: float = 0.0,
) -> str:

	api_key = _get_next_api_key()

	genai.configure(api_key=api_key)

	generative_model = genai.GenerativeModel(
		model_name=model,
		system_instruction=system_prompt,
	)

	generation_config = genai.types.GenerationConfig(
		temperature=temperature,
		# response_mime_type="application/json",
	)

	try:
		response = generative_model.generate_content(
			user_prompt,
			generation_config=generation_config,
		)

		text = (response.text or "").strip()

		if not text:
			raise LLMCallError(
				f"LLM returned empty response. Response: {response}"
			)
		return text
	
	except Exception as exc:
		raise LLMCallError(
			f"LLM call failed: {exc}"
		) from exc


def parse_json_response(raw_response: str) -> Any:

	cleaned = raw_response.strip()

	if cleaned.startswith("```"):
		cleaned = cleaned.strip("`")
		stripped_lower = cleaned.lstrip().lower()
		if stripped_lower.startswith("json"):
			cleaned = cleaned.lstrip()[4:]
		cleaned = cleaned.strip()

	try:
		return json.loads(cleaned)

	except json.JSONDecodeError as exc:
		raise LLMJSONParseError(
			f"Failed to parse LLM output as JSON: {exc}. "
			f"Raw output:\n{raw_response}"
		) from exc


# TESTING

# if __name__ == "__main__":

# 	# JSON test
# 	json_result = call_llm(
# 		system_prompt='Respond ONLY with JSON: {"answer":"<one word>"}.',
# 		user_prompt="What is the capital of France?",
# 	)

# 	print("\nRaw JSON:")
# 	print(json_result)

# 	print("\nParsed:")
# 	print(parse_json_response(json_result))