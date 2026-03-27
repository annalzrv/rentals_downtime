import json
import os
import sys
import importlib
import pytest
from pathlib import Path


def test_main_creates_report_and_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    (tmp_path / "data").mkdir(exist_ok=True)

    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main

    main.main()

    assert (tmp_path / "report.txt").exists()
    assert (tmp_path / "experiment_log.json").exists()

    with open(tmp_path / "experiment_log.json") as f:
        log = json.load(f)

    assert "timestamp" in log
    assert "duration_seconds" in log
    assert "total_input_tokens" in log
    assert "total_output_tokens" in log
    assert "final_mse" in log
    assert log["final_mse"] == 4230.5
    assert "iteration_count" in log
    assert "mse_history" in log
    assert log["mse_history"] == [4230.5]
    assert "supervisor_reasoning" in log
    assert "config" in log
    assert log["config"]["mock_llm"] is True

    with open(tmp_path / "report.txt") as f:
        report = f.read()
    assert "Total execution time" in report
    assert "Final MSE: 4230.5" in report