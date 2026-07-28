from pathlib import Path

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 5
MAX_STEPS = 4

AI_SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = AI_SERVICE_DIR.parent.parent

DEFAULT_INDEX_DIR = str(
    BACKEND_DIR / "knowledge-base" / "FAISS"
)

NO_EVIDENCE_MARKER = "No relevant evidence found"