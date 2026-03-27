"""
Shared state definition for the rentals_agents LangGraph pipeline.

THIS FILE IS THE TEAM CONTRACT — all nodes must honour these keys.

Field ownership:
  df_info           — Data_Profiler writes once at startup
  features_plan     — RAG_Domain_Expert writes; can be extended on re-entry
  generated_code    — Coder_Agent writes
  execution_result  — Code_Executor writes (stdout + stderr combined)
  execution_ok      — Code_Executor writes (True = no error, submission valid)
  metrics           — Code_Executor / MLOps module writes {"mse": float, ...}
  iteration_count   — Code_Executor increments by 1 per full cycle
  next_node         — Supervisor_Agent writes after JSON parse
  mse_history       — APPEND-ONLY via LangGraph reducer (do not overwrite)
  supervisor_reasoning — Supervisor_Agent writes (optional, for logs)
"""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

# ── Routing constants ─────────────────────────────────────────────────────────

NextNode = Literal["RAG_Domain_Expert", "Coder_Agent", "END"]

VALID_NEXT_NODES: frozenset[str] = frozenset(
    {"RAG_Domain_Expert", "Coder_Agent", "END"}
)


# ── State TypedDict ───────────────────────────────────────────────────────────

class State(TypedDict):
    # --- Data context ---
    df_info: str
    """Human-readable summary: dtypes, nulls, target stats. Written by Data_Profiler."""

    features_plan: list[str]
    """List of feature-engineering ideas. Written (and possibly extended) by RAG node."""

    # --- Code generation & execution ---
    generated_code: str
    """Raw Python source produced by Coder_Agent."""

    execution_result: str
    """Combined stdout + stderr from Code_Executor. Available to Supervisor."""

    execution_ok: bool
    """True if Code_Executor ran without error AND submission.csv passed validation."""

    # --- Metrics ---
    metrics: dict
    """At minimum: {"mse": float}. Written by MLOps module after each run."""

    # --- Loop control ---
    iteration_count: int
    """Incremented by Code_Executor once per full cycle."""

    # --- Routing ---
    next_node: str
    """Parsed from Supervisor JSON. Guardrails in route_after_supervisor may override."""

    # --- History (append-only via reducer) ---
    mse_history: Annotated[list[float], operator.add]
    """
    Accumulates MSE values across iterations.
    Nodes must return {"mse_history": [new_value]} — NOT the full list.
    LangGraph's operator.add reducer concatenates automatically.
    """

    # --- Debug / logging ---
    supervisor_reasoning: NotRequired[str]
    """Copy of the 'reasoning' field from Supervisor JSON. Optional."""


# ── Convenience: empty initial state ─────────────────────────────────────────

def initial_state() -> State:
    """Return a minimal valid starting state for graph.invoke()."""
    return State(
        df_info="",
        features_plan=[],
        generated_code="",
        execution_result="",
        execution_ok=False,
        metrics={},
        iteration_count=0,
        next_node="",
        mse_history=[],
    )
