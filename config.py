"""
Central configuration for the Multi-Agent Software Architect pipeline.
STATGR 5293 — Columbia University Spring 2026
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CORPUS_DIR      = BASE_DIR / "rag_corpus"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
HISTORY_DB      = str(BASE_DIR / "design_history.db")
OUTPUT_DIR      = BASE_DIR / "output"

# ── Embedding model (CPU-only, ~80 MB, free) ──────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_MODEL        = "gemini-2.5-flash"
LLM_TEMPERATURE     = 0.1
LLM_MAX_OUTPUT_TOKENS = 8192

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K = 6          # chunks retrieved per agent query

# ── Agent → vector store filter tags (must match build_vector_store.py) ───────
AGENT_TAGS = {
    "requirements": "requirements_agent",
    "architecture": "architecture_agent",
    "data_modeler": "data_modeler_agent",
    "critic":       "critic_agent",
    "diagram":      "diagram_agent",
}