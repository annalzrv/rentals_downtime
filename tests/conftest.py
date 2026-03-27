"""Test bootstrap for local `src/` imports without editable install."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rentals_agents.benchmark import Benchmark
from rentals_agents.rag.service import _get_retriever


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def reset_benchmark():
    Benchmark.reset()
    _get_retriever.cache_clear()
    yield
