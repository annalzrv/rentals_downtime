import os
import tempfile
import pytest
import pandas as pd
import rentals_agents.config as config
from rentals_agents.graph.nodes import executor_node
from rentals_agents.state import State

def test_executor_node_real(tmp_path, monkeypatch):
    # Создаём структуру данных
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))

    train_df = pd.DataFrame({
        "_id": [1, 2],
        "target": [100, 200],
        "sum": [150, 80],
    })
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df = pd.DataFrame({"_id": [3, 4]})
    test_df.to_csv(data_dir / "test.csv", index=False)
    sample_df = pd.DataFrame(columns=["index", "prediction"])
    sample_df.to_csv(data_dir / "sample_submition.csv", index=False)

    # Включаем реальный режим
    monkeypatch.setattr(config, "MOCK_LLM", False)

    # Код, который выводит MSE и создаёт submission.csv
    code = """
import pandas as pd
print("MSE: 123.45")
df = pd.DataFrame({"index": [0, 1], "prediction": [100.5, 200.5]})
df.to_csv("submission.csv", index=False)
"""

    state = State(
        df_info="",
        features_plan=[],
        generated_code=code,
        execution_result="",
        execution_ok=False,
        metrics={},
        iteration_count=0,
        next_node="",
        mse_history=[],
    )

    # Переходим во временную папку, чтобы все файлы создавались там
    monkeypatch.chdir(tmp_path)

    result = executor_node(state)

    assert result["execution_ok"] is True
    assert result["metrics"]["mse"] == 123.45
    assert result["mse_history"] == [123.45]
    assert result["iteration_count"] == 1
    assert "MSE: 123.45" in result["execution_result"]

    # Проверяем, что submission.csv создан
    submission_path = tmp_path / "submission.csv"
    assert submission_path.exists()
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["index", "prediction"]