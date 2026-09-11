import os
import pickle
import re
from pathlib import Path
from typing import Optional

import networkx as nx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_DIR: Path = Path(os.getenv("GRAPH_INDEX_DIR", str(BASE_DIR / "data" / "indexes" / "graph")))
GRAPH_FILE = "civic_ai_graph.pkl"


# ---------------------------------------------------------------------------
# Generic entity extraction patterns
# ---------------------------------------------------------------------------

# Named entities: proper nouns / titled phrases (capitalized multi-word sequences)
_NAMED_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,6})\b"
)

# Key concept signals: sentences that define, describe, or introduce a concept
_KEY_CONCEPT_PATTERN = re.compile(
    r"([^.]*?\b(?:refers to|is defined as|means|is described as|is known as|is called|is a|are a|denotes|represents|encompasses|comprises|includes)[^.]*\.)",
    re.IGNORECASE,
)

# Attribute / quantitative fact signals: numbers, measurements, percentages, dates
_ATTRIBUTE_PATTERN = re.compile(
    r"([^.]*?\b(?:\d[\d,\.]*\s*(?:%|percent|per cent|kg|km|mg|ml|years?|months?|days?|hours?|crore|lakh|million|billion|thousand)|"
    r"as of|as at|total of|up to|at least|no more than|minimum|maximum|approximately|around|nearly|"
    r"increased|decreased|reduced|improved|grew|declined)[^.]*\.)",
    re.IGNORECASE,
)

# Relationship / process signals: sentences describing how things interact or work
_RELATIONSHIP_PATTERN = re.compile(
    r"([^.]*?\b(?:depends on|leads to|results in|causes|enables|supports|requires|provides|responsible for|"
    r"implemented by|managed by|overseen by|funded by|administered by|linked to|associated with|"
    r"in order to|so that|therefore|consequently|thus|hence)[^.]*\.)",
    re.IGNORECASE,
)


def _extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract generic entities from chunk text using regex patterns.
    Returns a dict keyed by entity type with deduplicated value lists.
    Values are trimmed to 120 chars to keep node labels readable.

    Entity types:
      named_entity       — proper-noun multi-word titles/names found in the text
      key_concept        — sentences that define or describe a concept
      attribute          — sentences containing quantitative or measurable facts
      relationship       — sentences describing causal or structural relationships
    """
    entities: dict[str, list[str]] = {
        "named_entity": [], "key_concept": [], "attribute": [], "relationship": []
    }

    for m in _NAMED_ENTITY_PATTERN.finditer(text):
        val = m.group(1).strip()
        # Skip single-word matches and very long phrases that are likely sentence fragments
        if " " in val and len(val) <= 80 and val not in entities["named_entity"]:
            entities["named_entity"].append(val)

    for m in _KEY_CONCEPT_PATTERN.finditer(text):
        val = m.group(1).strip()[:120]
        if val not in entities["key_concept"]:
            entities["key_concept"].append(val)

    for m in _ATTRIBUTE_PATTERN.finditer(text):
        val = m.group(1).strip()[:120]
        if val not in entities["attribute"]:
            entities["attribute"].append(val)

    for m in _RELATIONSHIP_PATTERN.finditer(text):
        val = m.group(1).strip()[:120]
        if val not in entities["relationship"]:
            entities["relationship"].append(val)

    return entities


class GraphSearchResult:
    __slots__ = ("chunk_id", "score", "rank", "reason", "metadata")

    def __init__(self, chunk_id: str, score: float, rank: int, reason: str = "keyword_match", metadata: Optional[dict] = None) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.rank = rank
        self.reason = reason
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"GraphSearchResult(rank={self.rank}, score={self.score:.4f}, reason='{self.reason}', doc='{self.metadata.get('document_name','')}' p{self.metadata.get('page_number','?')})"


def build_graph(chunks: list) -> nx.DiGraph:
    G = nx.DiGraph()
    prev_chunk_per_page: dict[tuple, str] = {}
    section_chunks: dict[tuple, list[str]] = {}

    # Track named-entity nodes globally so multiple chunks referencing the same
    # proper noun link to one shared node rather than duplicates.
    entity_nodes: dict[str, str] = {}  # slug → entity_name

    for chunk in chunks:
        doc_id = chunk.document_id
        page_num = chunk.page_number
        section = chunk.section
        cid = chunk.chunk_id

        doc_node = f"doc::{doc_id}"
        page_node = f"page::{doc_id}::{page_num}"
        chunk_node = f"chunk::{cid}"
        section_slug = section.lower().replace(" ", "_")[:60] if section else None
        section_node = f"sec::{doc_id}::{section_slug}" if section_slug else None

        if not G.has_node(doc_node):
            G.add_node(doc_node, type="document", document_id=doc_id,
                       document_name=chunk.document_name, source=chunk.source)

        if not G.has_node(page_node):
            G.add_node(page_node, type="page", document_id=doc_id, page_number=page_num)
            G.add_edge(doc_node, page_node, rel="HAS_PAGE")

        if section_node and not G.has_node(section_node):
            G.add_node(section_node, type="section", document_id=doc_id, section=section)
            G.add_edge(page_node, section_node, rel="HAS_SECTION")

        G.add_node(chunk_node, type="chunk", chunk_id=cid, document_id=doc_id,
                   document_name=chunk.document_name, page_number=page_num, section=section,
                   domain=chunk.domain, source=chunk.source, word_count=chunk.word_count,
                   text=chunk.text)

        parent_node = section_node if section_node else page_node
        G.add_edge(parent_node, chunk_node, rel="HAS_CHUNK")

        # Sequential linking within a page
        page_key = (doc_id, page_num)
        if page_key in prev_chunk_per_page:
            G.add_edge(prev_chunk_per_page[page_key], chunk_node, rel="NEXT_CHUNK")
        prev_chunk_per_page[page_key] = chunk_node

        if section:
            section_chunks.setdefault((doc_id, section), []).append(chunk_node)

        # ── Entity extraction and knowledge graph nodes ────────────────────
        entities = _extract_entities(chunk.text)

        # Named entity nodes — shared globally by normalised name across all chunks/docs
        # so multiple chunks referencing the same proper noun converge on one node.
        for entity_name in entities["named_entity"]:
            slug = entity_name.lower().replace(" ", "_")[:80]
            node_id = f"entity::{slug}"
            if node_id not in entity_nodes:
                G.add_node(node_id, type="named_entity", name=entity_name,
                           domain=chunk.domain)
                entity_nodes[node_id] = entity_name
            G.add_edge(chunk_node, node_id, rel="MENTIONS_ENTITY")
            G.add_edge(node_id, chunk_node, rel="MENTIONED_IN")

        # Key concept nodes — scoped to this chunk (definitions / descriptions)
        for i, concept in enumerate(entities["key_concept"]):
            node_id = f"concept::{cid}::{i}"
            G.add_node(node_id, type="key_concept", description=concept,
                       document_id=doc_id, page_number=page_num)
            G.add_edge(chunk_node, node_id, rel="DEFINES_CONCEPT")

        # Attribute nodes — scoped to this chunk (quantitative / measurable facts)
        for i, attr in enumerate(entities["attribute"]):
            node_id = f"attribute::{cid}::{i}"
            G.add_node(node_id, type="attribute", value=attr,
                       document_id=doc_id, page_number=page_num)
            G.add_edge(chunk_node, node_id, rel="HAS_ATTRIBUTE")

        # Relationship nodes — scoped to this chunk (causal / structural links)
        for i, rel_text in enumerate(entities["relationship"]):
            node_id = f"relation::{cid}::{i}"
            G.add_node(node_id, type="relationship", description=rel_text,
                       document_id=doc_id, page_number=page_num)
            G.add_edge(chunk_node, node_id, rel="HAS_RELATIONSHIP")

    # SAME_SECTION edges between chunks sharing a section
    for _, nodes in section_chunks.items():
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                G.add_edge(a, b, rel="SAME_SECTION")
                G.add_edge(b, a, rel="SAME_SECTION")

    return G


def save_graph(G: nx.DiGraph, graph_dir: Path = DEFAULT_GRAPH_DIR) -> Path:
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / GRAPH_FILE
    with open(path, "wb") as f:
        pickle.dump(G, f)
    return path


def load_graph(graph_dir: Path = DEFAULT_GRAPH_DIR) -> nx.DiGraph:
    path = Path(graph_dir) / GRAPH_FILE
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as f:
        G: nx.DiGraph = pickle.load(f)
    return G


def _keyword_score(text: str, keywords: list[str]) -> float:
    if not keywords or not text:
        return 0.0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower) / len(keywords)


def keyword_search(
    G: nx.DiGraph,
    keywords: list[str],
    top_k: int = 5,
    domain_filter: Optional[str] = None,
    min_coverage_ratio: float = 0.3,
) -> list[GraphSearchResult]:
    if not keywords:
        return []

    min_hits = max(1, round(len(keywords) * min_coverage_ratio))
    results: list[tuple[float, str, dict]] = []

    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") != "chunk":
            continue
        if domain_filter and attrs.get("domain") != domain_filter:
            continue
        text_lower = attrs.get("text", "").lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits < min_hits:
            continue
        results.append((hits / len(keywords), node_id, attrs))

    results.sort(key=lambda x: x[0], reverse=True)
    return [
        GraphSearchResult(
            chunk_id=attrs["chunk_id"], score=score, rank=i + 1,
            reason="keyword_match",
            metadata={
                "document_name": attrs.get("document_name"), "page_number": attrs.get("page_number"),
                "section": attrs.get("section"), "domain": attrs.get("domain"),
                "source": attrs.get("source"), "text": attrs.get("text"),
            },
        )
        for i, (score, _, attrs) in enumerate(results[:top_k])
    ]


def get_related_chunks(
    G: nx.DiGraph,
    chunk_id: str,
    hops: int = 1,
    relation_filter: Optional[list[str]] = None,
) -> list[GraphSearchResult]:
    start_node = f"chunk::{chunk_id}"
    if not G.has_node(start_node):
        return []

    visited: set[str] = {start_node}
    frontier: set[str] = {start_node}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbour in G.successors(node):
                if neighbour in visited:
                    continue
                rel = G.get_edge_data(node, neighbour, default={}).get("rel", "")
                if relation_filter and rel not in relation_filter:
                    continue
                next_frontier.add(neighbour)
                visited.add(neighbour)
        frontier = next_frontier

    results: list[GraphSearchResult] = []
    for i, node in enumerate(visited - {start_node}):
        attrs = G.nodes[node]
        if attrs.get("type") != "chunk":
            continue
        results.append(GraphSearchResult(
            chunk_id=attrs["chunk_id"], score=1.0 / (i + 2), rank=0,
            reason="graph_traversal",
            metadata={
                "document_name": attrs.get("document_name"), "page_number": attrs.get("page_number"),
                "section": attrs.get("section"), "domain": attrs.get("domain"),
                "source": attrs.get("source"), "text": attrs.get("text"),
            },
        ))

    results.sort(key=lambda r: -r.score)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def get_section_chunks(G: nx.DiGraph, document_id: str, section: str) -> list[GraphSearchResult]:
    section_slug = section.lower().replace(" ", "_")[:60]
    section_node = f"sec::{document_id}::{section_slug}"

    if not G.has_node(section_node):
        return []

    results: list[GraphSearchResult] = []
    for neighbour in G.successors(section_node):
        attrs = G.nodes[neighbour]
        if attrs.get("type") != "chunk":
            continue
        results.append(GraphSearchResult(
            chunk_id=attrs["chunk_id"], score=1.0, rank=0, reason="section_match",
            metadata={
                "document_name": attrs.get("document_name"), "page_number": attrs.get("page_number"),
                "section": attrs.get("section"), "domain": attrs.get("domain"),
                "source": attrs.get("source"), "text": attrs.get("text"),
            },
        ))

    results.sort(key=lambda r: r.metadata.get("page_number", 0))
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def graph_stats(G: nx.DiGraph) -> dict:
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    for _, _, attrs in G.edges(data=True):
        r = attrs.get("rel", "unknown")
        edge_types[r] = edge_types.get(r, 0) + 1
    return {"total_nodes": G.number_of_nodes(), "total_edges": G.number_of_edges(),
            "node_types": node_types, "edge_types": edge_types}


def get_chunks_for_entity(G: nx.DiGraph, entity_name: str) -> list[GraphSearchResult]:
    """
    Return all chunk nodes linked to a named_entity node matching entity_name.
    Matches by substring on the node's 'name' attribute (case-insensitive).
    Used by multi-hop retrieval to follow entity → chunks edges.
    """
    results: list[GraphSearchResult] = []
    entity_name_lower = entity_name.lower()

    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") != "named_entity":
            continue
        if entity_name_lower not in attrs.get("name", "").lower():
            continue
        # Follow MENTIONED_IN edges back to chunk nodes
        for neighbour in G.successors(node_id):
            chunk_attrs = G.nodes[neighbour]
            if chunk_attrs.get("type") != "chunk":
                continue
            results.append(GraphSearchResult(
                chunk_id=chunk_attrs["chunk_id"],
                score=1.0,
                rank=0,
                reason="entity_lookup",
                metadata={
                    "document_name": chunk_attrs.get("document_name"),
                    "page_number": chunk_attrs.get("page_number"),
                    "section": chunk_attrs.get("section"),
                    "domain": chunk_attrs.get("domain"),
                    "source": chunk_attrs.get("source"),
                    "text": chunk_attrs.get("text"),
                },
            ))

    results.sort(key=lambda r: r.metadata.get("page_number", 0))
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def get_entity_nodes(G: nx.DiGraph, entity_type: str) -> list[dict]:
    """
    Return all nodes of a given entity type with their attributes.
    entity_type: one of 'named_entity', 'key_concept', 'attribute', 'relationship'
    """
    return [
        {"node_id": nid, **attrs}
        for nid, attrs in G.nodes(data=True)
        if attrs.get("type") == entity_type
    ]


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     from ingestion.parser import parse_all_pdfs
#     from ingestion.chunker import chunk_documents

#     PDF_DIR = BASE_DIR / "data" / "pdfs"
#     GRAPH_DIR = BASE_DIR / "data" / "indexes" / "graph"

#     print(f"\n{'='*60}")
#     print("GRAPH — Phase 2 test")
#     print(f"PDF dir: {PDF_DIR}\n")

#     documents = parse_all_pdfs(PDF_DIR)
#     if not documents:
#         print("No PDFs found.")
#         sys.exit(1)

#     chunks = chunk_documents(documents)
#     G = build_graph(chunks)
#     stats = graph_stats(G)
#     print(f"Nodes: {stats['total_nodes']}  Edges: {stats['total_edges']}")
#     print(f"Node types: {stats['node_types']}")
#     print(f"Edge types: {stats['edge_types']}")
#     save_graph(G, GRAPH_DIR)

#     G2 = load_graph(GRAPH_DIR)
#     for keywords in [["education", "scholarship", "student"], ["eligibility", "criteria", "benefit"]]:
#         min_hits = max(1, round(len(keywords) * 0.5))
#         print(f"\nKeywords: {keywords}  (need ≥{min_hits})")
#         results = keyword_search(G2, keywords, top_k=3, min_coverage_ratio=0.5)
#         for r in results:
#             print(f"  [{r.rank}] {r.score:.3f}  {r.metadata.get('document_name','?')} p{r.metadata.get('page_number','?')}")

#     # Show entity node counts
#     stats2 = graph_stats(G2)
#     print(f"\nEntity nodes extracted:")
#     for etype in ("scheme", "eligibility", "benefit", "document_requirement"):
#         count = stats2["node_types"].get(etype, 0)
#         print(f"  {etype:<22}: {count}")

#     # Show a sample scheme node and the chunks it links to
#     scheme_nodes = get_entity_nodes(G2, "scheme")
#     if scheme_nodes:
#         sample = scheme_nodes[0]
#         print(f"\nSample scheme node: '{sample.get('name')}'")
#         linked = get_chunks_for_scheme(G2, sample.get("name", ""))
#         print(f"  Linked chunks: {len(linked)}")
#         for r in linked[:2]:
#             print(f"    p{r.metadata.get('page_number','?')}: {r.metadata.get('text','')[:80]}...")

#     print("\nGraph test complete.")
