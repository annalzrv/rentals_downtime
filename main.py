"""
Entry point for the rentals_agents pipeline.

Usage:
    python main.py                  # mock mode (no Ollama needed)
    MOCK_LLM=0 python main.py       # real LLM mode (requires Ollama)
"""

import json
import time
from rentals_agents.graph.builder import build_graph
from rentals_agents.state import initial_state
from rentals_agents.benchmark import Benchmark
import rentals_agents.config as config


def main() -> None:
    benchmark = Benchmark()
    benchmark.start()

    graph = build_graph()
    result = graph.invoke(initial_state())

    benchmark.stop()

    if result.get("mse_history"):
        benchmark.set_final_mse(result["mse_history"][-1])
    elif result.get("metrics", {}).get("mse"):
        benchmark.set_final_mse(result["metrics"]["mse"])

    with open("report.txt", "w") as f:
        f.write(benchmark.report())

    log_data = {
        "timestamp": time.time(),
        "duration_seconds": benchmark.end_time - benchmark.start_time if benchmark.end_time and benchmark.start_time else 0,
        "total_input_tokens": benchmark.total_input_tokens,
        "total_output_tokens": benchmark.total_output_tokens,
        "final_mse": benchmark.final_mse,
        "iteration_count": result.get("iteration_count"),
        "mse_history": result.get("mse_history"),
        "supervisor_reasoning": result.get("supervisor_reasoning"),
        "config": {
            "target_mse_threshold": config.TARGET_MSE_THRESHOLD,
            "max_iterations": config.MAX_GRAPH_ITERATIONS,
            "mock_llm": config.MOCK_LLM,
            "data_dir": config.DATA_DIR,
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "qwen_coder_model": config.QWEN_CODER_MODEL,
            "llm_model": config.LLM_MODEL,
        },
    }
    with open("experiment_log.json", "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    print("\n=== Pipeline complete ===")
    print(f"Iterations   : {result['iteration_count']}")
    print(f"Final MSE    : {result['metrics'].get('mse', 'n/a')}")
    print(f"MSE history  : {result['mse_history']}")
    print(f"Supervisor   : {result.get('supervisor_reasoning', '')}")
    print("Report saved to report.txt")
    print("Experiment log saved to experiment_log.json")


if __name__ == "__main__":
    main()