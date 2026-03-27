"""Tests for vector RAG, source provenance, and prompt-quality helpers."""

import os

os.environ.setdefault("MOCK_LLM", "1")

from rentals_agents.rag.evaluation import build_rag_user_message, evaluate_feature_plan
from rentals_agents.rag.knowledge_base import load_source_documents
from rentals_agents.rag.service import retrieve_knowledge


def test_source_manifest_enriches_documents():
    documents = load_source_documents("data/knowledge_base")
    assert all(document.source_url for document in documents)
    assert any(document.source_kind == "catboost_docs" for document in documents)


def test_chroma_backend_can_run_offline_with_hash_embeddings():
    summary = (
        "Dataset has location_cluster, lat, lon, sum, last_dt, avg_reviews and total_host. "
        "Need top feature ideas for rental demand."
    )
    result = retrieve_knowledge(
        summary,
        top_k=3,
        backend="chroma",
        embedding_backend="hash",
    )

    assert result.backend == "chroma"
    assert len(result.chunks) == 3
    assert "url=" in result.context


def test_feature_plan_quality_gate_marks_good_plan_as_adequate():
    plan = [
        "dist_to_midtown_km from lat/lon for location premium",
        "review_weekday from last_dt for temporal demand effects",
        "log_sum from sum to reduce skew",
        "has_reviews from amt_reviews and avg_reviews",
        "host_portfolio_log from total_host",
    ]
    report = evaluate_feature_plan(plan)
    assert report.is_adequate is True
    assert report.total_covered >= 4


def test_build_rag_user_message_includes_retrieved_context():
    message = build_rag_user_message("df info", "retrieved snippets here")
    assert "Dataset description" in message
    assert "Retrieved domain context" in message
