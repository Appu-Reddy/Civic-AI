import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_call_count: int = 0


def _load_valid_keys() -> list[str]:
    candidates = [
        os.getenv("GEMINI_API_KEY_1", ""),
        os.getenv("GEMINI_API_KEY_2", ""),
        os.getenv("GEMINI_API_KEY_3", ""),
        os.getenv("GEMINI_API_KEY", ""),
    ]
    _placeholders = {
        "your_gemini_api_key_here", "your_gemini_api_key_1_here",
        "your_gemini_api_key_2_here", "your_gemini_api_key_3_here",
    }
    valid = [k.strip() for k in candidates if k.strip() and k.strip() not in _placeholders]
    seen: set[str] = set()
    unique: list[str] = []
    for k in valid:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def get_next_key() -> str:
    global _call_count
    keys = _load_valid_keys()
    if not keys:
        raise EnvironmentError("No Gemini API key found. Set GEMINI_API_KEY_1/2/3 in .env")
    key = keys[_call_count % len(keys)]
    _call_count += 1
    return key


def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    import google.generativeai as genai
    genai.configure(api_key=get_next_key())
    return genai.GenerativeModel(model_name)


def key_status() -> dict:
    keys = _load_valid_keys()
    total = len(keys)
    previews = [f"KEY_{i+1}: {k[:6]}...{k[-4:]}" if len(k) > 10 else f"KEY_{i+1}: ****" for i, k in enumerate(keys)]
    return {
        "total_keys": total,
        "call_count": _call_count,
        "next_key_index": (_call_count % total) + 1 if total else 0,
        "key_previews": previews,
    }