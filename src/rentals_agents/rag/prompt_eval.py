"""CLI-style helpers for evaluating the RAG prompt against real LLM outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from rentals_agents.config import LLM_MODEL
from rentals_agents.llm.json_utils import parse_json_response
from rentals_agents.llm.ollama_client import OllamaError, chat
from rentals_agents.prompts.system import RAG_SYSTEM_PROMPT
from rentals_agents.rag.evaluation import build_rag_user_message, evaluate_feature_plan
from rentals_agents.rag.service import retrieve_knowledge


@dataclass(frozen=True)
class PromptEvalCase:
    """One dataset-summary scenario for RAG prompt evaluation."""

    name: str
    dataset_summary: str


def default_eval_cases() -> list[PromptEvalCase]:
    """Return a small regression suite for the RAG prompt."""
    return [
        PromptEvalCase(
            name="balanced_airbnb_schema",
            dataset_summary=(
                "Columns: location_cluster, location, lat, lon, type_house, sum, "
                "min_days, amt_reviews, last_dt, avg_reviews, total_host, target. "
                "Need robust feature ideas for short-term rental demand prediction."
            ),
        ),
        PromptEvalCase(
            name="sparse_reviews_case",
            dataset_summary=(
                "Columns: borough, lat, lon, sum, amt_reviews, avg_reviews, last_dt. "
                "Many rows have missing avg_reviews and missing last_dt because new listings "
                "have no reviews yet."
            ),
        ),
    ]


def run_prompt_eval(model: str = LLM_MODEL) -> dict:
    """Run a lightweight real-LLM evaluation for the RAG node prompt."""
    results: list[dict] = []
    for case in default_eval_cases():
        retrieval = retrieve_knowledge(case.dataset_summary)
        user_message = build_rag_user_message(case.dataset_summary, retrieval.context)
        raw = chat(model, RAG_SYSTEM_PROMPT, user_message)
        parsed = parse_json_response(raw)
        ideas = parsed.get("ideas", [])
        report = evaluate_feature_plan(ideas)
        results.append(
            {
                "case": asdict(case),
                "backend": retrieval.backend,
                "top_sources": [hit.chunk.source_title for hit in retrieval.chunks],
                "ideas": ideas,
                "quality": asdict(report),
            }
        )

    passed = sum(1 for result in results if result["quality"]["is_adequate"])
    return {
        "model": model,
        "cases_run": len(results),
        "cases_passed": passed,
        "all_passed": passed == len(results),
        "results": results,
    }


def main() -> None:
    try:
        report = run_prompt_eval()
    except OllamaError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
