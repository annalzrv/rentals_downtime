"""Tests for prompt-evaluation helpers that do not require a live LLM."""

from rentals_agents.rag.prompt_eval import default_eval_cases


def test_default_prompt_eval_cases_are_defined():
    cases = default_eval_cases()
    assert len(cases) >= 2
    assert all(case.name for case in cases)
    assert all(case.dataset_summary for case in cases)
