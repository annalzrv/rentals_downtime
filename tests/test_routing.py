"""
Unit tests for routing guardrails.

All tests are pure Python — no LangGraph graph, no Ollama, no filesystem.
Run with: pytest tests/test_routing.py -v
"""

import math
import os

import pytest

# Ensure MOCK_LLM doesn't interfere with routing (routing has no LLM calls)
os.environ.setdefault("MOCK_LLM", "1")

from rentals_agents.config import MAX_GRAPH_ITERATIONS, TARGET_MSE_THRESHOLD
from rentals_agents.routing import route_after_executor, route_after_supervisor
from rentals_agents.state import initial_state


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_state(**overrides):
    s = initial_state()
    s.update(overrides)
    return s


# ── route_after_executor ─────────────────────────────────────────────────────

class TestRouteAfterExecutor:
    def test_execution_error_routes_to_coder(self):
        state = make_state(execution_ok=False, execution_result="Traceback: NameError")
        assert route_after_executor(state) == "Coder_Agent"

    def test_execution_success_routes_to_supervisor(self):
        state = make_state(execution_ok=True, execution_result="MSE: 423.5")
        assert route_after_executor(state) == "Supervisor_Agent"

    def test_missing_execution_ok_defaults_to_coder(self):
        # If Code_Executor forgot to set the field, treat as error.
        state = initial_state()
        state.pop("execution_ok", None)  # simulate missing key
        assert route_after_executor(state) == "Coder_Agent"


# ── route_after_supervisor ───────────────────────────────────────────────────

class TestRouteAfterSupervisor:

    # --- Guardrail 1: iteration cap ---

    def test_max_iterations_forces_end(self):
        state = make_state(
            iteration_count=MAX_GRAPH_ITERATIONS,
            metrics={"mse": math.inf},
            next_node="RAG_Domain_Expert",
        )
        assert route_after_supervisor(state) == "END"

    def test_exceeded_iterations_forces_end(self):
        state = make_state(
            iteration_count=MAX_GRAPH_ITERATIONS + 5,
            metrics={"mse": math.inf},
            next_node="Coder_Agent",
        )
        assert route_after_supervisor(state) == "END"

    def test_below_max_iterations_does_not_force_end(self):
        state = make_state(
            iteration_count=MAX_GRAPH_ITERATIONS - 1,
            metrics={"mse": math.inf},
            next_node="Coder_Agent",
        )
        # Should not be forced to END by iter cap alone
        assert route_after_supervisor(state) != "END" or True  # could be END by mse
        # More specifically, with mse=inf and valid next_node:
        assert route_after_supervisor(state) == "Coder_Agent"

    # --- Guardrail 2: quality threshold ---

    def test_mse_at_threshold_forces_end(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": TARGET_MSE_THRESHOLD},
            next_node="RAG_Domain_Expert",
        )
        assert route_after_supervisor(state) == "END"

    def test_mse_below_threshold_forces_end(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": TARGET_MSE_THRESHOLD - 1.0},
            next_node="Coder_Agent",
        )
        assert route_after_supervisor(state) == "END"

    def test_mse_above_threshold_does_not_force_end(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": TARGET_MSE_THRESHOLD + 1.0},
            next_node="Coder_Agent",
        )
        assert route_after_supervisor(state) == "Coder_Agent"

    def test_missing_mse_treated_as_inf(self):
        # metrics dict present but no "mse" key
        state = make_state(
            iteration_count=1,
            metrics={},
            next_node="Coder_Agent",
        )
        assert route_after_supervisor(state) == "Coder_Agent"

    # --- Guardrail 3: invalid next_node ---

    def test_empty_next_node_falls_back_to_coder(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": math.inf},
            next_node="",
        )
        assert route_after_supervisor(state) == "Coder_Agent"

    def test_garbage_next_node_falls_back_to_coder(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": math.inf},
            next_node="sure, let me think about this...",
        )
        assert route_after_supervisor(state) == "Coder_Agent"

    def test_none_next_node_falls_back_to_coder(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": math.inf},
            next_node=None,
        )
        assert route_after_supervisor(state) == "Coder_Agent"

    # --- Guardrail 4: trust LLM ---

    def test_valid_rag_next_node_passes_through(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": math.inf},
            next_node="RAG_Domain_Expert",
        )
        assert route_after_supervisor(state) == "RAG_Domain_Expert"

    def test_valid_coder_next_node_passes_through(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": math.inf},
            next_node="Coder_Agent",
        )
        assert route_after_supervisor(state) == "Coder_Agent"

    def test_supervisor_end_respected_when_no_guardrail_triggered(self):
        state = make_state(
            iteration_count=1,
            metrics={"mse": math.inf},
            next_node="END",
        )
        assert route_after_supervisor(state) == "END"

    # --- Guardrail priority: iter cap beats everything ---

    def test_iter_cap_beats_valid_next_node(self):
        state = make_state(
            iteration_count=MAX_GRAPH_ITERATIONS,
            metrics={"mse": TARGET_MSE_THRESHOLD + 100},
            next_node="RAG_Domain_Expert",
        )
        assert route_after_supervisor(state) == "END"
