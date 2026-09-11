from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvidenceRef:
    chunk_id: str
    document_name: str
    page_number: int
    section: Optional[str]
    text: str
    source: str


@dataclass
class HistoryEntry:
    step_number: int
    objective: str
    summary: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    satisfied: bool = True

    def source_citations(self) -> list[str]:
        seen: set[str] = set()
        citations: list[str] = []
        for ev in self.evidence:
            cite = f"{ev.document_name} p{ev.page_number}"
            if cite not in seen:
                seen.add(cite)
                citations.append(cite)
        return citations

    def to_context_block(self) -> str:
        lines = [f"[Step {self.step_number}] {self.objective}", f"Summary: {self.summary}"]
        for entity_type, values in self.entities.items():
            if values:
                lines.append(f"{entity_type}: {', '.join(values)}")
        citations = self.source_citations()
        if citations:
            lines.append(f"Sources: {'; '.join(citations)}")
        if not self.satisfied:
            lines.append("Note: Insufficient evidence found for this step.")
        return "\n".join(lines)


class HistoryStore:
    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []

    def add(self, entry: HistoryEntry) -> None:
        self._entries.append(entry)

    def get(self, step_number: int) -> Optional[HistoryEntry]:
        for entry in self._entries:
            if entry.step_number == step_number:
                return entry
        return None

    def all_entries(self) -> list[HistoryEntry]:
        return sorted(self._entries, key=lambda e: e.step_number)

    def all_evidence(self) -> list[EvidenceRef]:
        seen: set[str] = set()
        evidence: list[EvidenceRef] = []
        for entry in self._entries:
            for ev in entry.evidence:
                if ev.chunk_id not in seen:
                    seen.add(ev.chunk_id)
                    evidence.append(ev)
        return evidence

    def all_entities(self) -> dict[str, list[str]]:
        merged: dict[str, set[str]] = {}
        for entry in self._entries:
            for entity_type, values in entry.entities.items():
                merged.setdefault(entity_type, set()).update(values)
        return {k: sorted(v) for k, v in merged.items()}

    def covers_objective(self, objective: str, threshold: float = 0.6) -> bool:
        obj_words = set(objective.lower().split())
        for entry in self._entries:
            if not entry.satisfied:
                continue
            hist_words = set(entry.objective.lower().split())
            if not obj_words or not hist_words:
                continue
            overlap = len(obj_words & hist_words) / len(obj_words | hist_words)
            if overlap >= threshold:
                return True
        return False

    def to_context_string(self) -> str:
        if not self._entries:
            return ""
        blocks = [e.to_context_block() for e in self.all_entries()]
        return "--- Prior Reasoning Steps ---\n" + "\n\n".join(blocks) + "\n--- End ---"

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def step_count(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"HistoryStore({len(self._entries)} entries)"