"""
LangGraph graph builder for the rentals_agents pipeline.

Usage:
    from rentals_agents.graph.builder import build_graph
    graph = build_graph()
    result = graph.invoke(initial_state())
"""

from langgraph.graph import END, StateGraph

from rentals_agents.graph.nodes import (
    coder_node,
    data_profiler_node,
    executor_node,
    rag_node,
    supervisor_node,
)
from rentals_agents.routing import route_after_executor, route_after_supervisor
from rentals_agents.state import State


def build_graph():
    """
    Compile and return the full LangGraph pipeline.

    Graph topology (mirrors team_plan.md):

        Data_Profiler → RAG_Domain_Expert → Coder_Agent → Code_Executor
                                                               │
                                              ┌─── error ─────┘
                                              ▼
                                         Coder_Agent
                                              │
                                   success ──┘
                                              ▼
                                       Supervisor_Agent
                                              │
                             ┌───────────────┼───────────────┐
                             ▼               ▼               ▼
                     RAG_Domain_Expert  Coder_Agent         END
    """
    g = StateGraph(State)

    # Register nodes
    g.add_node("Data_Profiler", data_profiler_node)
    g.add_node("RAG_Domain_Expert", rag_node)
    g.add_node("Coder_Agent", coder_node)
    g.add_node("Code_Executor", executor_node)
    g.add_node("Supervisor_Agent", supervisor_node)

    # Linear start: Data_Profiler → RAG → Coder → Executor
    g.set_entry_point("Data_Profiler")
    g.add_edge("Data_Profiler", "RAG_Domain_Expert")
    g.add_edge("RAG_Domain_Expert", "Coder_Agent")
    g.add_edge("Coder_Agent", "Code_Executor")

    # Branching 1 (after Code_Executor): deterministic, no LLM
    g.add_conditional_edges(
        "Code_Executor",
        route_after_executor,
        {
            "Coder_Agent": "Coder_Agent",
            "Supervisor_Agent": "Supervisor_Agent",
        },
    )

    # Branching 2 (after Supervisor): agentic + guardrails
    g.add_conditional_edges(
        "Supervisor_Agent",
        route_after_supervisor,
        {
            "RAG_Domain_Expert": "RAG_Domain_Expert",
            "Coder_Agent": "Coder_Agent",
            "END": END,
        },
    )

    return g.compile()
