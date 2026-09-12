"""
domain_index.py -- Build and persist the domain keyword map

Derives a keyword -> domain_slug mapping automatically from:
  1. Named-entity nodes in the knowledge graph (high-quality multi-word phrases
     already extracted at ingest time, scoped to their source document domain)
  2. Chunk text token frequency per domain (words that appear in many chunks of
     one domain but few of another are strong domain signals)

The result is saved as data/indexes/domain_keywords.json with this schema:
{
    "india_nep": ["education", "curriculum", "foundational literacy", ...],
    "india_nhp": ["health", "hospital", "primary care", ...]
}

Each value list is deduplicated and sorted by signal strength (most
discriminative first).

Public API
----------
    build_domain_keywords(graph, chunk_metadata) -> dict[str, list[str]]
    save_domain_keywords(mapping, path)
    load_domain_keywords(path) -> dict[str, list[str]]
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DOMAIN_KEYWORDS_FILE = "domain_keywords.json"

_STOPWORDS: frozenset = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "that", "this", "these", "those", "with", "from", "by", "at", "on",
    "as", "it", "its", "into", "through", "about", "over", "under", "after",
    "before", "between", "also", "more", "such", "not", "but", "they",
    "their", "them", "there", "here", "which", "when", "where", "how",
    "what", "who", "why", "all", "each", "every", "any", "some", "than",
    "then", "than", "both", "other", "can", "well", "per",
})


def _tokenise(text: str, min_len: int = 4) -> list:
    """Lowercase single tokens, stripped of punctuation, above min_len."""
    tokens = []
    for word in text.lower().split():
        word = word.strip(".,;:!?\"'()-/\\")
        if len(word) >= min_len and word not in _STOPWORDS:
            tokens.append(word)
    return tokens


def _slug_from_document_name(document_name: str) -> str:
    """Normalise a document name to a slug (e.g. 'India_NEP' -> 'india_nep')."""
    return re.sub(r"[^a-z0-9]+", "_", document_name.lower()).strip("_")


def build_domain_keywords(
    graph,
    chunk_metadata: dict,
    min_chunk_count: int = 2,
    max_keywords_per_domain: int = 120,
    entity_boost: int = 5,
) -> dict:
    """
    Build a keyword -> domain_slug mapping from the graph and chunk metadata.

    Parameters
    ----------
    graph : nx.DiGraph
        The loaded knowledge graph (from retrieval.graph.load_graph).
    chunk_metadata : dict
        The FAISS chunk metadata map {chunk_id: {document_name, domain, text, ...}}.
    min_chunk_count : int
        A token must appear in at least this many chunks of a domain to be kept.
    max_keywords_per_domain : int
        Maximum keywords to keep per domain (most frequent first).
    entity_boost : int
        Named-entity phrase occurrences are multiplied by this factor so
        multi-word phrases rank above single noisy tokens.

    Returns
    -------
    dict[str, list[str]]
        {domain_slug: [keyword, ...]} sorted by frequency descending.
    """
    # domain_slug -> Counter{keyword: chunk_count}
    domain_token_counts: dict = defaultdict(Counter)
    # domain_slug -> set of chunks seen (for per-chunk counting)
    domain_chunk_sets: dict = defaultdict(set)

    # ── Pass 1: chunk text tokens from graph nodes ────────────────────────────
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "chunk":
            continue
        domain = attrs.get("domain", "").strip()
        if not domain:
            # Derive slug from document_name as fallback
            domain = _slug_from_document_name(attrs.get("document_name", "unknown"))
        text = attrs.get("text", "")
        chunk_id = attrs.get("chunk_id", node_id)

        if chunk_id in domain_chunk_sets[domain]:
            continue  # already counted this chunk
        domain_chunk_sets[domain].add(chunk_id)

        for token in set(_tokenise(text)):  # set: count each token once per chunk
            domain_token_counts[domain][token] += 1

    # ── Pass 2: chunk text tokens from FAISS metadata (covers any gaps) ──────
    for chunk_id, meta in chunk_metadata.items():
        domain = meta.get("domain", "").strip()
        if not domain:
            domain = _slug_from_document_name(meta.get("document_name", "unknown"))
        if chunk_id in domain_chunk_sets[domain]:
            continue
        domain_chunk_sets[domain].add(chunk_id)
        text = meta.get("text", "")
        for token in set(_tokenise(text)):
            domain_token_counts[domain][token] += 1

    # ── Pass 3: named-entity phrases from graph (boosted weight) ─────────────
    # Each named_entity node has a 'domain' attribute from the chunk it was
    # extracted from and a 'name' attribute that is a clean multi-word phrase.
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "named_entity":
            continue
        domain = attrs.get("domain", "").strip()
        if not domain:
            continue
        phrase = attrs.get("name", "").strip().lower()
        if len(phrase) >= 4 and phrase not in _STOPWORDS:
            domain_token_counts[domain][phrase] += entity_boost

        # Also add individual tokens of the phrase
        for token in _tokenise(phrase):
            domain_token_counts[domain][token] += entity_boost

    # ── Build final mapping ───────────────────────────────────────────────────
    # Count how many domains each keyword appears in
    kw_domain_count: dict = {}
    for domain, counter in domain_token_counts.items():
        for kw in counter:
            kw_domain_count[kw] = kw_domain_count.get(kw, 0) + 1

    mapping: dict = {}
    for domain, counter in domain_token_counts.items():
        qualified = [
            (kw, count) for kw, count in counter.items()
            if count >= min_chunk_count
            and len(kw) >= 4
            and kw_domain_count.get(kw, 1) == 1  # unique to this domain
        ]
        qualified.sort(key=lambda x: -x[1])
        mapping[domain] = [kw for kw, _ in qualified[:max_keywords_per_domain]]

    total_kw = sum(len(v) for v in mapping.values())
    logger.info(
        "build_domain_keywords: %d domains, %d total keywords",
        len(mapping), total_kw,
    )
    return mapping


def save_domain_keywords(mapping: dict, index_dir: Path) -> Path:
    """Save the mapping to <index_dir>/domain_keywords.json."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / DOMAIN_KEYWORDS_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.info("Saved domain keywords")
    return path


def load_domain_keywords(index_dir: Path) -> Optional[dict]:
    """
    Load domain_keywords.json from index_dir.
    Returns None if the file does not exist (triggers fallback in step_definer).
    """
    path = Path(index_dir) / DOMAIN_KEYWORDS_FILE
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    logger.info("Loaded domain keywords from")
    return mapping