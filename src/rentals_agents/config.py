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

# ── Development / CI mode ────────────────────────────────────────────────────
# MOCK_LLM=1 (default): nodes return deterministic stubs — no Ollama needed.
# MOCK_LLM=0: nodes call the real Ollama server.
MOCK_LLM: bool = os.getenv("MOCK_LLM", "1") == "1"
