"""
LangGraph node functions for the rentals_agents pipeline.

Each node:
  - Accepts `state: State`
  - Returns a `dict` with ONLY the keys it updates (LangGraph merges the rest)
  - Is safe to run with MOCK_LLM=True (no real Ollama / no real dataset needed)

MOCK_LLM mode (default, MOCK_LLM=1):
  Nodes return deterministic stubs so the graph can be tested in CI without
  Ollama and without the Kaggle dataset.

Real mode (MOCK_LLM=0):
  Nodes call ollama_client.chat() with the proper system prompts.
  Requires Ollama running at OLLAMA_BASE_URL with the configured models.

Teammates: replace mock logic inside each function with your real implementation.
Keep the function signature and return-key contract unchanged.
"""

import csv
import io
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import rentals_agents.config as config
from rentals_agents.llm.json_utils import parse_json_response
from rentals_agents.llm.ollama_client import OllamaError, chat
from rentals_agents.prompts.system import (
    CODER_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    supervisor_system_prompt,
)
from rentals_agents.rag import (
    build_rag_user_message,
    evaluate_feature_plan,
    generate_mock_feature_plan,
    retrieve_knowledge,
)
from rentals_agents.state import State

import subprocess
import tempfile
import os
import re
import sys
import pandas as pd

_MSE_LINE_RE = re.compile(
    r"(?im)^\s*MSE:\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_SUBMISSION_COLUMNS = ("index", "prediction")
_EXECUTOR_TIMEOUT_SECONDS = 300.0


def _extract_mse_from_output(output: str) -> float:
    match = _MSE_LINE_RE.search(output)
    if match is None:
        return float("inf")
    return float(match.group(1))


def _validate_submission_csv(path: Path) -> str | None:
    if not path.exists():
        return "submission.csv is missing"

    try:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if tuple(reader.fieldnames or ()) != _SUBMISSION_COLUMNS:
                return (
                    "submission.csv must have exactly columns "
                    "'index,prediction' in this order"
                )

            has_rows = False
            for row in reader:
                has_rows = True
                raw_index = row.get("index")
                raw_prediction = row.get("prediction")

                if raw_index is None or raw_prediction is None:
                    return "submission.csv contains malformed rows"

                try:
                    int(raw_index)
                    prediction = float(raw_prediction)
                except (TypeError, ValueError):
                    return "submission.csv must contain numeric index and prediction"

                if not math.isfinite(prediction):
                    return "submission.csv contains non-finite prediction values"

            if not has_rows:
                return "submission.csv is empty"
    except (OSError, csv.Error) as exc:
        return f"Failed to read submission.csv: {exc}"

    return None

# ── 1. Data_Profiler ──────────────────────────────────────────────────────────

def data_profiler_node(state: State) -> dict:
    """
    Load dataset and produce a text summary for downstream agents.

    Real implementation (Borya's module):
      - pd.read_csv("train.csv")
      - df.info(), df.describe(), target distribution, outlier count
      - Write summary string to df_info

    Mock: returns a static description of a plausible rental dataset.
    """
    if not config.MOCK_LLM:

        train_path = f"{config.DATA_DIR}/train.csv"
        df = pd.read_csv(train_path)

        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()

        target_stats = df["target"].describe().to_string()
        missing = df.isnull().sum()
        missing_str = missing[missing > 0].to_string() if missing.any() else "none"
        type_counts = df["type_house"].value_counts().to_string()
        borough_counts = df["location_cluster"].value_counts().to_string()

        summary = (
            f"=== df.info() ===\n{info_str}\n"
            f"=== target stats ===\n{target_stats}\n"
            f"=== missing values ===\n{missing_str}\n"
            f"=== type_house distribution ===\n{type_counts}\n"
            f"=== location_cluster (borough) distribution ===\n{borough_counts}\n"
            f"=== sum (listed price/night) range ===\n"
            f"min={df['sum'].min()}, max={df['sum'].max()}, mean={df['sum'].mean():.1f}\n"
            f"Data path: {train_path}"
        )
        return {"df_info": summary}

    mock_df_info = (
        "Dataset: NYC short-term rental listings — Airbnb-style (train.csv)\n"
        "Rows: 36 671  |  Columns: 15\n"
        "Target: target (float64, mean=112.8, std=131.6, min=0, max=365)\n"
        "Columns: name, _id, host_name, location_cluster, location, lat, lon,\n"
        "         type_house, sum, min_days, amt_reviews, last_dt,\n"
        "         avg_reviews, total_host, target\n"
        "Key columns:\n"
        "  location_cluster — NYC borough: Manhattan (44%), Brooklyn (41%),\n"
        "                     Queens (12%), Bronx (2%), Staten Island (1%)\n"
        "  type_house       — Entire home/apt (52%), Private room (46%), Shared room (2%)\n"
        "  sum              — listed price per night, range $0–$10 000, mean $152\n"
        "  last_dt          — date of last review (NaN if no reviews yet)\n"
        "  lat, lon         — GPS coordinates (NYC range)\n"
        "Missing values: last_dt (20.5%), avg_reviews (20.5%), name (0.02%), host_name (0.05%)\n"
        "Note: test.csv has same columns minus 'target'; submission format: index,prediction"
    )
    return {"df_info": mock_df_info}


# ── 2. RAG_Domain_Expert ──────────────────────────────────────────────────────

def rag_node(state: State) -> dict:
    """
    Generate feature-engineering ideas using RAG + LLM.

    Real implementation (RAG engineer's module):
      - Query ChromaDB with df_info to retrieve relevant chunks
      - Pass chunks + df_info to LLM as context
      - Parse JSON {"ideas": [...]}

    Mock: returns a hardcoded, realistic feature plan.
    """
    if not config.MOCK_LLM:
        user_msg = (
            f"Dataset description:\n{state['df_info']}\n\n"
            "Based on this dataset, suggest feature engineering ideas for "
            "predicting rental price per night."
        )
    retrieval = retrieve_knowledge(state["df_info"])

    if not MOCK_LLM:
        user_msg = build_rag_user_message(state["df_info"], retrieval.context)
        try:
            raw = chat(config.LLM_MODEL, RAG_SYSTEM_PROMPT, user_msg)
            parsed = parse_json_response(raw)
            ideas: list[str] = parsed.get("ideas", [])
            report = evaluate_feature_plan(ideas)
            if not report.is_adequate:
                ideas = generate_mock_feature_plan(state["df_info"])
        except (OllamaError, ValueError) as exc:
            # Degrade gracefully: return a minimal fallback plan
            ideas = [f"[RAG error — using fallback] {exc}"]
            ideas.extend(generate_mock_feature_plan(state["df_info"]))
        return {"features_plan": ideas}

    return {"features_plan": generate_mock_feature_plan(state["df_info"])}


# ── 3. Coder_Agent ────────────────────────────────────────────────────────────

def coder_node(state: State) -> dict:
    """
    Generate Python code for feature engineering + model training.

    The generated code must:
      - Use TimeSeriesSplit for cross-validation
      - Print "MSE: <float>" to stdout (executor parses this)
      - Write submission.csv

    Mock: returns a minimal valid Python script that prints a fixed MSE.
    """
    if not config.MOCK_LLM:
        error_context = ""
        if state.get("execution_result") and not state.get("execution_ok", True):
            error_context = (
                f"\n\nPrevious execution error (fix this):\n{state['execution_result']}"
            )

        plan_str = "\n".join(f"- {idea}" for idea in state.get("features_plan", []))
        user_msg = (
            f"Dataset description:\n{state['df_info']}\n\n"
            f"Feature ideas to implement:\n{plan_str}"
            f"{error_context}"
        )
        try:
            raw = chat(config.QWEN_CODER_MODEL, CODER_SYSTEM_PROMPT, user_msg)
            parsed = parse_json_response(raw)
            code: str = parsed.get("code", "")
        except (OllamaError, ValueError) as exc:
            code = f'raise RuntimeError("Coder failed to generate code: {exc}")'
        return {"generated_code": code}

    # Mock: a trivial script that satisfies the executor's MSE-parsing contract.
    mock_code = '''\
# Mock training script — replace with real implementation
mse = 4230.5
print(f"MSE: {mse}")

# Write a dummy submission.csv (format: index,prediction)
with open("submission.csv", "w") as f:
    f.write("index,prediction\\n")
    for i in range(10):
        f.write(f"{i},112.8\\n")
'''
    return {"generated_code": mock_code}


# ── 4. Code_Executor ─────────────────────────────────────────────────────────

def executor_node(state: State) -> dict:
    """
    Run the generated Python code in an isolated subprocess.

    Real implementation (DevOps module):
      - Write generated_code to a temp file
      - subprocess.run() with timeout and restricted env
      - Parse stdout for "MSE: <float>"
      - Validate submission.csv (no NaN, correct columns)
      - Set execution_ok and execution_result accordingly

    Mock: simulates a successful run and extracts the hardcoded MSE from mock code.
    """
    new_iteration_count = state.get("iteration_count", 0) + 1

    if not config.MOCK_LLM:
        code = state.get("generated_code", "")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd()
            )
            stdout = result.stdout
            stderr = result.stderr
            execution_result = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            execution_ok = (result.returncode == 0)

            mse_match = re.search(r"MSE:\s*([\d.]+)", stdout)
            if mse_match:
                mse_value = float(mse_match.group(1))
            else:
                mse_value = None
                execution_ok = False
                execution_result += "\n[ERROR] MSE not found in output"

            submission_path = "submission.csv"
            if os.path.exists(submission_path):
                df_sub = pd.read_csv(submission_path)
                required_cols = ['index', 'prediction']
                if not all(col in df_sub.columns for col in required_cols):
                    execution_ok = False
                    execution_result += "\n[ERROR] submission.csv missing required columns"
                elif df_sub.isnull().any().any():
                    execution_ok = False
                    execution_result += "\n[ERROR] submission.csv contains NaN"
            else:
                execution_ok = False
                execution_result += "\n[ERROR] submission.csv not found"

            metrics = {"mse": mse_value} if mse_value is not None else {}

        except subprocess.TimeoutExpired:
            execution_result = "Timeout (60s) while running script"
            execution_ok = False
            metrics = {}
        except Exception as e:
            execution_result = f"Execution error: {e}"
            execution_ok = False
            metrics = {}
        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)

        return {
            "execution_result": execution_result,
            "execution_ok": execution_ok,
            "metrics": metrics,
            "mse_history": [metrics.get("mse")] if metrics else [],
            "iteration_count": new_iteration_count,
            "supervisor_reasoning": "",
    if not MOCK_LLM:
        generated_code = state.get("generated_code", "")
        if not generated_code.strip():
            msg = "No generated code found."
            return {
                "execution_result": msg,
                "execution_ok": False,
                "metrics": {"mse": float("inf")},
                "mse_history": [float("inf")],
                "iteration_count": new_iteration_count,
            }

        work_dir = Path(os.getcwd())
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_script = Path(tmp_dir) / "agent_script.py"
            temp_script.write_text(generated_code, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, str(temp_script)],
                    cwd=str(work_dir),
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=_EXECUTOR_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "execution_result": (
                        f"Code execution timed out after "
                        f"{_EXECUTOR_TIMEOUT_SECONDS} seconds."
                    ),
                    "execution_ok": False,
                    "metrics": {"mse": float("inf")},
                    "mse_history": [float("inf")],
                    "iteration_count": new_iteration_count,
                }

        output = (result.stdout or "").strip()
        if result.stderr:
            stderr_part = (result.stderr or "").strip()
            if output:
                output = f"{output}\n--- stderr ---\n{stderr_part}"
            else:
                output = stderr_part

        mse = _extract_mse_from_output((result.stdout or ""))
        is_mse_finite = math.isfinite(mse)
        if not is_mse_finite:
            mse_for_state = float("inf")
        else:
            mse_for_state = mse

        if result.returncode != 0:
            return {
                "execution_result": output,
                "execution_ok": False,
                "metrics": {"mse": mse_for_state},
                "mse_history": [mse_for_state],
                "iteration_count": new_iteration_count,
            }

        if not is_mse_finite:
            return {
                "execution_result": (
                    f"{output}\nValidation error: output has no valid MSE line"
                    if output
                    else "Validation error: output has no valid MSE line"
                ),
                "execution_ok": False,
                "metrics": {"mse": float("inf")},
                "mse_history": [float("inf")],
                "iteration_count": new_iteration_count,
            }

        submission_error = _validate_submission_csv(work_dir / "submission.csv")
        if submission_error is not None:
            return {
                "execution_result": (
                    f"{output}\nValidation error: {submission_error}"
                    if output
                    else f"Validation error: {submission_error}"
                ),
                "execution_ok": False,
                "metrics": {"mse": mse_for_state},
                "mse_history": [mse_for_state],
                "iteration_count": new_iteration_count,
            }

        return {
            "execution_result": output,
            "execution_ok": True,
            "metrics": {"mse": mse_for_state},
            "mse_history": [mse_for_state],
            "iteration_count": new_iteration_count,
        }

    # Mock: extract MSE from the generated code string (works for mock_code above)
    code = state.get("generated_code", "")
    mse_match = re.search(r"mse\s*=\s*([\d.]+)", code)
    mse_value = float(mse_match.group(1)) if mse_match else 999.0

    mock_stdout = f"MSE: {mse_value}\nsubmission.csv written (10 rows, format: index,prediction)"
    metrics = {"mse": mse_value}

    return {
        "execution_result": mock_stdout,
        "execution_ok": True,
        "metrics": metrics,
        "mse_history": [mse_value],          # reducer appends this
        "iteration_count": new_iteration_count,
        "supervisor_reasoning": "",
    }


# ── 5. Supervisor_Agent ───────────────────────────────────────────────────────

def supervisor_node(state: State) -> dict:
    """
    Evaluate the run result and choose the next step.

    Writes next_node and supervisor_reasoning to state.
    Actual routing guardrails are applied AFTER this node in route_after_supervisor.

    Mock: returns "END" unconditionally so smoke tests terminate quickly.
    """
    if not config.MOCK_LLM:
        system = supervisor_system_prompt(config.TARGET_MSE_THRESHOLD, config.MAX_GRAPH_ITERATIONS)
        mse_history = state.get("mse_history", [])
        user_msg = (
            f"mse_history: {mse_history}\n"
            f"iteration_count: {state.get('iteration_count', 0)}\n"
            f"execution_result (last 500 chars): "
            f"{state.get('execution_result', '')[-500:]}\n"
            f"features_plan (summary): "
            f"{'; '.join(state.get('features_plan', [])[:3])} ..."
        )
        try:
            raw = chat(config.LLM_MODEL, system, user_msg)
            parsed = parse_json_response(raw)
            next_node: str = parsed.get("next_node", "")
            reasoning: str = parsed.get("reasoning", "")
        except (OllamaError, ValueError) as exc:
            # On parse failure, write empty next_node; guardrails will fall back.
            next_node = ""
            reasoning = f"[parse error: {exc}]"
        return {"next_node": next_node, "supervisor_reasoning": reasoning}

    return {
        "next_node": "END",
        "supervisor_reasoning": "Mock supervisor: terminating after first iteration.",
    }
