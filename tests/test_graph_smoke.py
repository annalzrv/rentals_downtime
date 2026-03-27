"""
Smoke test: full graph run with MOCK_LLM=1.

Verifies that:
  - The graph compiles without errors
  - A complete invoke() call terminates (reaches END)
  - iteration_count is incremented
  - mse_history contains at least one entry
  - execution_ok is set
  - supervisor_reasoning is populated
"""

import os

import pytest

# Must be set before importing config (module-level constant)
os.environ["MOCK_LLM"] = "1"

pytest.importorskip("langgraph")

from rentals_agents.graph.builder import build_graph
from rentals_agents.state import initial_state


@pytest.fixture(scope="module")
def compiled_graph():
    return build_graph()


@pytest.fixture(scope="module")
def run_result(compiled_graph):
    state = initial_state()
    return compiled_graph.invoke(state)


class TestGraphCompilation:
    def test_graph_compiles(self, compiled_graph):
        assert compiled_graph is not None

    def test_graph_has_nodes(self, compiled_graph):
        # LangGraph exposes node names via get_graph()
        node_names = set(compiled_graph.get_graph().nodes.keys())
        expected = {
            "Data_Profiler",
            "RAG_Domain_Expert",
            "Coder_Agent",
            "Code_Executor",
            "Supervisor_Agent",
        }
        assert expected.issubset(node_names)


class TestGraphRun:
    def test_run_completes(self, run_result):
        assert run_result is not None

    def test_df_info_populated(self, run_result):
        assert isinstance(run_result["df_info"], str)
        assert len(run_result["df_info"]) > 10

    def test_features_plan_populated(self, run_result):
        plan = run_result["features_plan"]
        assert isinstance(plan, list)
        assert len(plan) >= 1

    def test_generated_code_populated(self, run_result):
        code = run_result["generated_code"]
        assert isinstance(code, str)
        assert len(code) > 10

    def test_execution_ok(self, run_result):
        assert run_result["execution_ok"] is True

    def test_iteration_count_incremented(self, run_result):
        assert run_result["iteration_count"] >= 1

    def test_mse_history_appended(self, run_result):
        history = run_result["mse_history"]
        assert isinstance(history, list)
        assert len(history) >= 1
        assert all(isinstance(v, float) for v in history)

    def test_metrics_has_mse(self, run_result):
        assert "mse" in run_result["metrics"]
        assert isinstance(run_result["metrics"]["mse"], float)

    def test_supervisor_reasoning_populated(self, run_result):
        reasoning = run_result.get("supervisor_reasoning", "")
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_next_node_is_valid_or_end(self, run_result):
        # After graph completes, next_node should be END (mock supervisor)
        assert run_result["next_node"] == "END"
