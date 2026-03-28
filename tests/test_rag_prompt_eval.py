"""Tests for prompt-evaluation helpers that do not require a live LLM."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rentals_agents.llm.ollama_client import OllamaError
from rentals_agents.rag import prompt_eval


def test_default_prompt_eval_cases_are_defined():
    cases = prompt_eval.default_eval_cases()
    assert len(cases) >= 2
    assert all(case.name for case in cases)
    assert all(case.dataset_summary for case in cases)


def test_run_prompt_eval_returns_structured_report(monkeypatch):
    monkeypatch.setattr(
        prompt_eval,
        "default_eval_cases",
        lambda: [
            prompt_eval.PromptEvalCase(name="case_a", dataset_summary="summary a"),
            prompt_eval.PromptEvalCase(name="case_b", dataset_summary="summary b"),
        ],
    )

    def fake_retrieve(summary: str):
        return SimpleNamespace(
            backend="chroma",
            context=f"context for {summary}",
            chunks=[
                SimpleNamespace(chunk=SimpleNamespace(source_title="Time Series Feature Engineering")),
                SimpleNamespace(chunk=SimpleNamespace(source_title="CatBoost Practical Notes")),
            ],
        )

    responses = iter(
        [
            json.dumps(
                {
                    "ideas": [
                        "dist_to_midtown_km from lat/lon for location premium",
                        "review_weekday from last_dt for temporal demand effects",
                        "log_sum from sum to reduce skew",
                        "has_reviews from amt_reviews and avg_reviews",
                        "host_portfolio_log from total_host",
                    ]
                }
            ),
            json.dumps(
                {
                    "ideas": [
                        "borough_room_interaction from location_cluster and type_house",
                        "days_since_last_review from last_dt",
                        "log_sum from sum",
                        "has_reviews from amt_reviews",
                        "host_portfolio_log from total_host",
                    ]
                }
            ),
        ]
    )

    monkeypatch.setattr(prompt_eval, "retrieve_knowledge", fake_retrieve)
    monkeypatch.setattr(prompt_eval, "chat", lambda *args, **kwargs: next(responses))

    report = prompt_eval.run_prompt_eval(model="fake-model")

    assert report["model"] == "fake-model"
    assert report["cases_run"] == 2
    assert report["cases_passed"] == 2
    assert report["all_passed"] is True
    assert len(report["results"]) == 2
    assert report["results"][0]["backend"] == "chroma"
    assert report["results"][0]["top_sources"] == [
        "Time Series Feature Engineering",
        "CatBoost Practical Notes",
    ]
    assert report["results"][0]["quality"]["is_adequate"] is True


def test_run_prompt_eval_counts_failed_cases(monkeypatch):
    monkeypatch.setattr(
        prompt_eval,
        "default_eval_cases",
        lambda: [
            prompt_eval.PromptEvalCase(name="good_case", dataset_summary="summary 1"),
            prompt_eval.PromptEvalCase(name="bad_case", dataset_summary="summary 2"),
        ],
    )
    monkeypatch.setattr(
        prompt_eval,
        "retrieve_knowledge",
        lambda summary: SimpleNamespace(
            backend="lexical",
            context=f"context for {summary}",
            chunks=[SimpleNamespace(chunk=SimpleNamespace(source_title="Any Source"))],
        ),
    )

    responses = iter(
        [
            json.dumps(
                {
                    "ideas": [
                        "dist_to_midtown_km from lat/lon",
                        "review_weekday from last_dt",
                        "log_sum from sum",
                        "has_reviews from amt_reviews",
                        "host_portfolio_log from total_host",
                    ]
                }
            ),
            json.dumps({"ideas": ["encode location_cluster", "fill avg_reviews with 0"]}),
        ]
    )
    monkeypatch.setattr(prompt_eval, "chat", lambda *args, **kwargs: next(responses))

    report = prompt_eval.run_prompt_eval(model="fake-model")

    assert report["cases_run"] == 2
    assert report["cases_passed"] == 1
    assert report["all_passed"] is False
    assert report["results"][1]["quality"]["is_adequate"] is False


def test_main_prints_error_json_and_exits(monkeypatch, capsys):
    def broken_run() -> dict:
        raise OllamaError("cannot reach local ollama")

    monkeypatch.setattr(prompt_eval, "run_prompt_eval", broken_run)

    with pytest.raises(SystemExit) as exc_info:
        prompt_eval.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "cannot reach local ollama" in payload["error"]
