"""
step_definer.py -- Step enrichment

For each step dict ({"sub_query": "..."}), extracts relevant domain slugs
from the sub_query text and adds them to a "domains" list.

Domain slugs are document identifiers used by the graph retriever
(e.g. "india_nep", "india_nhp").  The keyword->slug mapping is built
automatically at ingest time and persisted to:

    data/indexes/domain_keywords.json

On first use this module loads that file and inverts it into a fast
{keyword: domain_slug} lookup table.  If the file does not exist yet
(indexes not built), it falls back to scanning the FAISS chunk_metadata_map
directly so the pipeline is never blocked.

The map is refreshed automatically on the next ingest run -- no code
changes are ever needed when new PDFs are added.

Public API
----------
    define_step(step: dict)  -> dict   (adds "domains" key)
    define_steps(steps: list) -> list
    reload_domain_map()               (force refresh, e.g. after re-ingest)
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


_BASE_DIR   = Path(__file__).resolve().parent.parent
_INDEX_DIR  = Path(os.getenv("FAISS_INDEX_DIR", str(_BASE_DIR / "data" / "indexes" / "faiss")))
_DOMAIN_KW_FILE  = _INDEX_DIR / "domain_keywords.json"
_METADATA_FILE   = _INDEX_DIR / "chunk_metadata_map.pkl"


_STOPWORDS: frozenset = frozenset({
    "what", "which", "when", "where", "who", "how", "why", "the", "a",
    "an", "are", "can", "for", "and", "that", "this", "with", "from",
    "have", "does", "will", "been", "some", "also", "more", "such",
    "was", "were", "has", "had", "not", "but", "its", "their", "about",
    "over", "under", "after", "before", "between", "there", "here",
    "they", "them", "these", "those", "into", "through", "given",
    "explain", "describe", "list", "tell", "find", "show", "compare",
    "summarise", "summarize", "according",
})


# {keyword_or_phrase: domain_slug}
_KW_TO_DOMAIN: dict = {}
_map_loaded: bool = False


def _slug_from_document_name(document_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", document_name.lower()).strip("_")


def _load_from_json() -> Optional[dict]:
    """Load and invert domain_keywords.json -> {keyword: domain_slug}."""
    if not _DOMAIN_KW_FILE.exists():
        return None
    import json
    with open(_DOMAIN_KW_FILE, "r", encoding="utf-8") as f:
        mapping: dict = json.load(f)  # {domain_slug: [keywords]}
    inverted: dict = {}
    for domain_slug, keywords in mapping.items():
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in inverted:
                inverted[kw_lower] = domain_slug
    # logger.info(
    #     "step_definer: loaded %d keywords across %d domains from %s",
    #     len(inverted), len(mapping), _DOMAIN_KW_FILE,
    # )
    return inverted


def _load_from_metadata() -> Optional[dict]:
    """
    Fallback: scan FAISS chunk_metadata_map.pkl directly.
    Tokenises every chunk's text and maps high-frequency tokens to
    their domain slug.  Less precise than the JSON file but always
    available if the indexes exist.
    """
    if not _METADATA_FILE.exists():
        return None
    import pickle
    from collections import Counter, defaultdict

    with open(_METADATA_FILE, "rb") as f:
        chunk_metadata: dict = pickle.load(f)

    domain_token_counts: dict = defaultdict(Counter)
    domain_chunk_sets: dict = defaultdict(set)

    for chunk_id, meta in chunk_metadata.items():
        domain = meta.get("domain", "").strip()
        if not domain:
            domain = _slug_from_document_name(meta.get("document_name", "unknown"))
        if chunk_id in domain_chunk_sets[domain]:
            continue
        domain_chunk_sets[domain].add(chunk_id)
        text = meta.get("text", "")
        for word in set(_tokenise(text)):
            domain_token_counts[domain][word] += 1

    inverted: dict = {}
    for domain, counter in domain_token_counts.items():
        for token, count in counter.most_common(120):
            if count >= 2 and token not in inverted:
                inverted[token] = domain

    logger.info(
        "step_definer: built %d keywords from metadata fallback (%d domains)",
        len(inverted), len(domain_token_counts),
    )
    return inverted


def _ensure_map_loaded() -> None:
    """Populate _KW_TO_DOMAIN on first use (lazy load)."""
    global _KW_TO_DOMAIN, _map_loaded
    if _map_loaded:
        return

    result = _load_from_json()
    if result is None:
        logger.warning(
            "step_definer: %s not found, falling back to metadata scan.",
            _DOMAIN_KW_FILE,
        )
        result = _load_from_metadata()

    if result is None:
        logger.warning(
            "step_definer: no domain map available -- domain detection disabled."
        )
        result = {}

    _KW_TO_DOMAIN = result
    _map_loaded = True


def reload_domain_map() -> None:
    """Force reload of the domain map (call after re-ingest)."""
    global _map_loaded
    _map_loaded = False
    _ensure_map_loaded()
    logger.info("Step_definer: domain map reloaded (%d keywords)", len(_KW_TO_DOMAIN))


def _tokenise(text: str, min_len: int = 4) -> list:
    tokens = []
    for word in text.lower().split():
        word = word.strip(".,;:!?\"'()-/\\")
        if len(word) >= min_len and word not in _STOPWORDS:
            tokens.append(word)
    return tokens


def _extract_domains(sub_query: str) -> list:
    """
    Match the sub_query against the loaded keyword map and return a
    deduplicated list of matching domain slugs.

    Multi-word phrases are checked first (longest-match wins) so that
    "higher education" maps to india_nep rather than just "education".
    Then single tokens are checked for any remaining coverage.
    """
    _ensure_map_loaded()
    if not _KW_TO_DOMAIN:
        return []

    text_lower = sub_query.lower()
    found: dict = {}  # ordered set via insertion-order dict

    # Multi-word phrase pass (phrases with spaces, longest first)
    phrases = sorted(
        (kw for kw in _KW_TO_DOMAIN if " " in kw),
        key=len,
        reverse=True,
    )
    for phrase in phrases:
        if phrase in text_lower:
            found[_KW_TO_DOMAIN[phrase]] = None

    # Single-token pass
    for token in _tokenise(sub_query):
        if token in _KW_TO_DOMAIN:
            found[_KW_TO_DOMAIN[token]] = None

    return list(found.keys())


def define_step(step: dict) -> dict:
    """
    Enrich a single step dict with extracted domain slugs.

    Input:
        {"sub_query": "What are the goals of NEP 2020?"}

    Output:
        {"sub_query": "What are the goals of NEP 2020?", "domains": ["india_nep"]}
    """
    sub_query = step.get("sub_query", "").strip()
    domains = _extract_domains(sub_query)
    enriched = {**step, "domains": domains}
    logger.debug("define_step: %r -> domains=%s", sub_query[:60], domains)
    return enriched


def define_steps(steps: list) -> list:
    """
    Enrich every step in the list with domain slugs.

    Input:
        [{"sub_query": "..."}, ...]

    Output:
        [{"sub_query": "...", "domains": [...]}, ...]
    """
    return [define_step(s) for s in steps]
