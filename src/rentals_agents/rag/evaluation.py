"""Evaluation helpers for RAG prompt quality and feature-plan adequacy."""

from __future__ import annotations

from dataclasses import dataclass


REQUIRED_SIGNAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "location": ("location", "borough", "cluster", "lat", "lon", "distance"),
    "time": ("last_dt", "time", "month", "weekday", "week", "recency", "lag"),
    "price": ("sum", "price", "log_sum", "ratio", "min_day"),
    "reviews": ("review", "amt_reviews", "avg_reviews", "has_reviews"),
    "host": ("host", "total_host", "portfolio"),
}


@dataclass(frozen=True)
class FeaturePlanReport:
    """Coverage summary for a generated feature plan."""

    covered_signals: dict[str, bool]
    total_covered: int
    is_adequate: bool


def build_rag_user_message(dataset_summary: str, retrieved_context: str) -> str:
    """Create the user message passed to the RAG LLM."""
    return (
        f"Dataset description:\n{dataset_summary}\n\n"
        f"Retrieved domain context:\n{retrieved_context}\n\n"
        "Based on this dataset, suggest feature engineering ideas for "
        "predicting rental availability / downtime."
    )


def evaluate_feature_plan(ideas: list[str], *, min_signals: int = 4) -> FeaturePlanReport:
    """Check whether a feature plan covers the main signal families we care about."""
    joined = " ".join(idea.lower() for idea in ideas)
    covered = {
        signal: any(keyword in joined for keyword in keywords)
        for signal, keywords in REQUIRED_SIGNAL_KEYWORDS.items()
    }
    total_covered = sum(1 for is_present in covered.values() if is_present)
    return FeaturePlanReport(
        covered_signals=covered,
        total_covered=total_covered,
        is_adequate=total_covered >= min_signals and len(ideas) >= 5,
    )
