"""
Entry point for the rentals_agents pipeline.

Usage:
    python main.py                  # mock mode (no Ollama needed)
    MOCK_LLM=0 python main.py       # real LLM mode (requires Ollama)
"""

from rentals_agents.graph.builder import build_graph
from rentals_agents.state import initial_state


def main() -> None:
    graph = build_graph()
    result = graph.invoke(initial_state())

    print("\n=== Pipeline complete ===")
    print(f"Iterations   : {result['iteration_count']}")
    print(f"Final MSE    : {result['metrics'].get('mse', 'n/a')}")
    print(f"MSE history  : {result['mse_history']}")
    print(f"Supervisor   : {result.get('supervisor_reasoning', '')}")


if __name__ == "__main__":
    main()
