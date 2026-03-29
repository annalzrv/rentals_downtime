"""
Routing functions with guardrails for the rentals_agents LangGraph pipeline.

Both functions are pure Python — no LLM calls, no IO.
They are registered as conditional edge functions in graph/builder.py.

Design principles
-----------------
* route_after_executor  — deterministic branch on execution outcome.
* route_after_supervisor — trusts Supervisor's JSON decision ONLY after all
  guardrails pass; guardrails are never bypassed by LLM output.
"""

import math
from typing import Literal

from rentals_agents.config import MAX_GRAPH_ITERATIONS, TARGET_MSE_THRESHOLD
from rentals_agents.state import VALID_NEXT_NODES, State


_MAX_CONSECUTIVE_ERRORS = 3


def route_after_executor(
    state: State,
) -> Literal["Coder_Agent", "Supervisor_Agent"]:
    """
    Deterministic branch after Code_Executor (no LLM involved).

    - execution_ok is True                          →  Supervisor_Agent
    - execution_ok is False, streak < 3             →  Coder_Agent (fix the code)
    - execution_ok is False, streak >= 3            →  Supervisor_Agent
      (let Supervisor decide: new features via RAG, different approach, or END)
    """
    if state.get("execution_ok", False):
        return "Supervisor_Agent"
    if state.get("consecutive_errors", 0) >= _MAX_CONSECUTIVE_ERRORS:
        return "Supervisor_Agent"
    return "Coder_Agent"


def route_after_supervisor(
    state: State,
) -> Literal["RAG_Domain_Expert", "Coder_Agent", "END"]:
    """
    Agentic routing after Supervisor_Agent, with Python-enforced guardrails.

    Guardrail priority (highest first):
    1. Iteration cap — force END if iteration_count >= MAX_GRAPH_ITERATIONS.
    2. Quality met   — force END if mse <= TARGET_MSE_THRESHOLD.
    3. Invalid route — fallback to Coder_Agent if next_node is not in whitelist
                       or is empty (e.g. JSON parse failure in supervisor node).
    4. Trust LLM     — pass through the Supervisor's decision otherwise.
    """
    iteration_count: int = state.get("iteration_count", 0)
    mse: float = state.get("metrics", {}).get("mse", math.inf)
    next_node: str = state.get("next_node", "")

    # Guardrail 1: hard iteration cap
    if iteration_count >= MAX_GRAPH_ITERATIONS:
        return "END"

    # Guardrail 2: quality threshold reached
    if mse <= TARGET_MSE_THRESHOLD:
        return "END"

    # Guardrail 3: invalid / missing next_node from LLM
    if next_node not in VALID_NEXT_NODES or next_node == "END":
        # "END" is valid for the supervisor to request, but if it hasn't been
        # triggered by guardrails 1/2 and the LLM says END, we still respect it.
        if next_node == "END":
            return "END"
        # Truly invalid → safe fallback
        return "Coder_Agent"

    # Guardrail 4: trust Supervisor
    return next_node  # type: ignore[return-value]
