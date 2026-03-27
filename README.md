# rentals_agents

LangGraph multi-agent pipeline for rental price prediction — HSE Agents course project.

## Architecture

```mermaid
flowchart LR
  DataProfiler[Data_Profiler]
  RAG[RAG_Domain_Expert]
  Coder[Coder_Agent]
  Exec[Code_Executor]
  Sup[Supervisor_Agent]
  EndNode[END]

  DataProfiler --> RAG --> Coder --> Exec
  Exec -->|error| Coder
  Exec -->|success| Sup
  Sup -->|route_after_supervisor| RAG
  Sup -->|route_after_supervisor| Coder
  Sup -->|route_after_supervisor| EndNode
```

### Nodes

| Node | Type | Owner |
|------|------|-------|
| `Data_Profiler` | Python function | MLOps (Borya) |
| `RAG_Domain_Expert` | LLM (Llama/Mistral) | RAG engineer |
| `Coder_Agent` | LLM (Qwen2.5-Coder) | Architect (Anna) |
| `Code_Executor` | Python + subprocess | DevOps |
| `Supervisor_Agent` | LLM (Llama/Mistral) | Architect (Anna) |

### Agentic routing & guardrails

After `Code_Executor`: **deterministic** branch — error → `Coder_Agent`, success → `Supervisor_Agent`.

After `Supervisor_Agent`: LLM proposes `next_node`, then **`route_after_supervisor`** applies guardrails in priority order:

1. `iteration_count >= MAX_GRAPH_ITERATIONS` → force **END**
2. `mse <= TARGET_MSE_THRESHOLD` → force **END**
3. invalid / unparseable `next_node` → fallback to **`Coder_Agent`**
4. Otherwise, trust the Supervisor's decision

The LLM **never** has final say on stopping — Python guardrails always run last.

---

## Quick start

### 1. Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

### 2. Run tests (no Ollama needed)

```bash
pytest tests/ -v
```

`MOCK_LLM=1` is the default. All tests pass without a running Ollama server.

### 3. Run with real models

Install Ollama: https://ollama.com

```bash
ollama pull qwen2.5-coder:7b
ollama pull llama3:8b

MOCK_LLM=0 python main.py
```

---

## Configuration

All settings via environment variables (no `.env` file required):

| Variable | Default | Description |
|----------|---------|-------------|
| `MOCK_LLM` | `1` | `1` = mock mode (no Ollama); `0` = real LLM calls |
| `DATA_DIR` | `data` | Directory with train.csv / test.csv (relative to project root) |
| `TARGET_MSE` | `500.0` | Stop when MSE ≤ this value |
| `MAX_ITER` | `10` | Hard iteration cap |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `QWEN_CODER_MODEL` | `qwen2.5-coder:7b` | Model for Coder_Agent |
| `LLM_MODEL` | `llama3:8b` | Model for RAG + Supervisor |
| `OLLAMA_TIMEOUT` | `120.0` | Request timeout (seconds) |

---

## Dataset

**Source:** MWS AI Agents 2026 Kaggle competition — NYC short-term rentals (Airbnb-style).

| Column | Description |
|--------|-------------|
| `name` | Listing title |
| `_id` | Unique listing ID |
| `host_name` | Host name |
| `location_cluster` | NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| `location` | Neighbourhood name |
| `lat`, `lon` | GPS coordinates |
| `type_house` | Entire home/apt · Private room · Shared room |
| `sum` | Listed price per night ($) |
| `min_days` | Minimum booking duration (days) |
| `amt_reviews` | Number of reviews |
| `last_dt` | Date of last review (NaN if no reviews) |
| `avg_reviews` | Average review score (NaN when amt_reviews = 0) |
| `total_host` | Number of listings by this host |
| `target` | **Target variable** (float, 0–365) |

Train: 36,671 rows. Test: 12,264 rows. Submission: `index,prediction`.

---

## Project structure

```
src/rentals_agents/
  config.py          # env-based constants
  state.py           # TypedDict State — team contract
  routing.py         # route_after_executor, route_after_supervisor (guardrails)
  llm/
    ollama_client.py # HTTP client for Ollama
    json_utils.py    # parse LLM JSON responses
  prompts/
    system.py        # system prompts for RAG, Coder, Supervisor
  graph/
    nodes.py         # node functions (mock + real stubs)
    builder.py       # StateGraph wiring
tests/
  test_routing.py    # guardrail unit tests (16 cases)
  test_graph_smoke.py  # end-to-end mock graph run
```

## For teammates

### Implementing your module

All nodes share the same contract: accept `state: State`, return `dict` with updated keys only.

**DevOps (Code_Executor):** replace `executor_node` in `graph/nodes.py`. Your function must set:
```python
return {
    "execution_result": "<stdout+stderr>",
    "execution_ok": True | False,
    "metrics": {"mse": <float>},
    "mse_history": [<float>],       # append ONE value; reducer accumulates
    "iteration_count": state["iteration_count"] + 1,
}
```

**RAG engineer (RAG_Domain_Expert):** replace `rag_node`. Your function must set:
```python
return {"features_plan": ["idea 1", "idea 2", ...]}
```

**MLOps/Borya (Data_Profiler + metrics):**
- Replace `data_profiler_node` — set `{"df_info": "<text summary>"}`.
- After your CV code runs, write MSE to `metrics` and append to `mse_history` in `executor_node`.
- `target_threshold` comes from `config.TARGET_MSE_THRESHOLD`; pass it to Supervisor prompt via `supervisor_system_prompt()`.

### State fields reference

See `src/rentals_agents/state.py` for the full TypedDict with field-level docstrings.

Key note: **`mse_history` uses a LangGraph reducer** (`operator.add`). Always return `{"mse_history": [new_value]}` — never the full list — or you will get double-appending.
