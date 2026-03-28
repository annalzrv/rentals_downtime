"""Tests for the local RAG knowledge base and retrieval layer."""

import os

os.environ.setdefault("MOCK_LLM", "1")

from rentals_agents.graph.nodes import rag_node
from rentals_agents.state import initial_state
from rentals_agents.rag import generate_mock_feature_plan, retrieve_knowledge
from rentals_agents.rag.knowledge_base import build_knowledge_chunks, load_source_documents


def test_knowledge_base_documents_load():
    documents = load_source_documents("data/knowledge_base")
    assert len(documents) >= 4
    assert any("CatBoost" in document.title for document in documents)


def test_chunk_builder_creates_chunks():
    documents = load_source_documents("data/knowledge_base")
    chunks = build_knowledge_chunks(documents, chunk_size=400, chunk_overlap=80)
    assert len(chunks) >= len(documents)
    assert all(chunk.text for chunk in chunks)


def test_retrieval_returns_time_series_context():
    summary = (
        "Columns include last_dt, location_cluster, lat, lon, sum, avg_reviews, total_host. "
        "Need robust cross-validation for rental demand over time."
    )
    result = retrieve_knowledge(summary, top_k=2)

    assert len(result.chunks) == 2
    assert "Time Series" in result.context or "TimeSeriesSplit" in result.context


def test_mock_feature_plan_uses_dataset_signals():
    summary = (
        "Columns: location_cluster, type_house, lat, lon, sum, min_days, amt_reviews, "
        "last_dt, avg_reviews, total_host"
    )
    plan = generate_mock_feature_plan(summary)

    assert len(plan) >= 5
    assert any("last_dt" in idea or "review" in idea.lower() for idea in plan)
    assert any("location_cluster" in idea or "borough" in idea.lower() for idea in plan)


def test_rag_node_in_mock_mode_does_not_call_retrieval(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("retrieve_knowledge should not be called in MOCK_LLM mode")

    monkeypatch.setattr("rentals_agents.graph.nodes.MOCK_LLM", True)
    monkeypatch.setattr("rentals_agents.graph.nodes.retrieve_knowledge", fail)

    state = initial_state()
    state["df_info"] = (
        "Columns: location_cluster, type_house, lat, lon, sum, min_days, "
        "amt_reviews, last_dt, avg_reviews, total_host"
    )

    result = rag_node(state)
    assert len(result["features_plan"]) >= 5
