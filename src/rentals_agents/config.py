"""
Central configuration for rentals_agents.

All values can be overridden via environment variables — no .env file required.
"""

import os

# ── Quality / stopping criteria ──────────────────────────────────────────────
# Supervisor will be told this value; guardrails enforce hard stop when met.
TARGET_MSE_THRESHOLD: float = float(os.getenv("TARGET_MSE", "500.0"))

# Hard cap on graph iterations regardless of MSE.
MAX_GRAPH_ITERATIONS: int = int(os.getenv("MAX_ITER", "10"))

# ── Ollama / vLLM backend ────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model used by Coder_Agent (code generation).
QWEN_CODER_MODEL: str = os.getenv("QWEN_CODER_MODEL", "qwen2.5-coder:7b")

# Model used by RAG_Domain_Expert and Supervisor_Agent (reasoning / text).
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3:8b")

# Request timeout in seconds for Ollama API calls.
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

# ── Dataset path ─────────────────────────────────────────────────────────────
# Directory containing train.csv, test.csv, sample_submition.csv
# Default: data/ relative to the project root (where you run pytest / main.py from).
# Override: DATA_DIR=/absolute/path python main.py
DATA_DIR: str = os.getenv("DATA_DIR", "data")

# ── RAG knowledge base ───────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "900"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RAG_MAX_CONTEXT_CHARS: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
RAG_RETRIEVER_BACKEND: str = os.getenv("RAG_RETRIEVER_BACKEND", "auto")
RAG_EMBEDDING_BACKEND: str = os.getenv("RAG_EMBEDDING_BACKEND", "auto")
RAG_EMBEDDING_CACHE_DIR: str = os.getenv("RAG_EMBEDDING_CACHE_DIR", ".cache/chroma")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rentals_feature_ideas")

# ── Development / CI mode ────────────────────────────────────────────────────
# MOCK_LLM=1 (default): nodes return deterministic stubs — no Ollama needed.
# MOCK_LLM=0: nodes call the real Ollama server.
MOCK_LLM: bool = os.getenv("MOCK_LLM", "1") == "1"
