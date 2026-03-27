"""High-level RAG service used by the `RAG_Domain_Expert` node."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rentals_agents.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    KNOWLEDGE_BASE_DIR,
    RAG_EMBEDDING_BACKEND,
    RAG_EMBEDDING_CACHE_DIR,
    RAG_RETRIEVER_BACKEND,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOP_K,
)
from rentals_agents.rag.knowledge_base import build_knowledge_chunks, load_source_documents
from rentals_agents.rag.vector_store import (
    ChromaVectorRetriever,
    HashEmbeddingBackend,
    OnnxMiniLMEmbeddingBackend,
)
from rentals_agents.rag.retriever import LexicalRetriever, ScoredChunk


@dataclass(frozen=True)
class RetrievalResult:
    """Top-ranked chunks and the prompt-ready context assembled from them."""

    chunks: list[ScoredChunk]
    context: str
    backend: str


@lru_cache(maxsize=8)
def _get_retriever(
    backend: str,
    embedding_backend: str,
):
    documents = load_source_documents(KNOWLEDGE_BASE_DIR)
    chunks = build_knowledge_chunks(
        documents,
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    resolved_backend = _resolve_backend(backend)
    if resolved_backend == "chroma":
        return ChromaVectorRetriever(
            chunks,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_backend=_resolve_embedding_backend(embedding_backend),
        )
    return LexicalRetriever(chunks)


def retrieve_knowledge(
    dataset_summary: str,
    *,
    top_k: int | None = None,
    backend: str | None = None,
    embedding_backend: str | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant knowledge-base chunks for the current dataset."""
    query = (
        "short term rental price prediction time series feature engineering "
        "catboost cross validation outliers availability reviews host location "
        f"{dataset_summary}"
    )
    final_k = top_k or RAG_TOP_K
    candidate_k = max(final_k * 3, 6)
    requested_backend = backend or RAG_RETRIEVER_BACKEND
    requested_embedding_backend = embedding_backend or RAG_EMBEDDING_BACKEND

    try:
        retriever = _get_retriever(requested_backend, requested_embedding_backend)
        effective_backend = _resolve_backend(requested_backend)
        if effective_backend == "chroma":
            vector_hits = retriever.search(query, top_k=candidate_k)
            lexical_hits = _get_retriever("lexical", "hash").search(query, top_k=candidate_k)
            hits = _hybrid_rerank(query, vector_hits, lexical_hits, final_k=final_k)
        else:
            hits = retriever.search(query, top_k=final_k)
    except Exception:
        retriever = _get_retriever("lexical", "hash")
        hits = retriever.search(query, top_k=final_k)
        effective_backend = "lexical"

    return RetrievalResult(
        chunks=hits,
        context=build_rag_prompt_context(hits),
        backend=effective_backend,
    )


def build_rag_prompt_context(chunks: list[ScoredChunk]) -> str:
    """Format retrieved chunks into a compact prompt section for the LLM."""
    if not chunks:
        return "No external knowledge retrieved. Fall back to robust baseline features."

    parts: list[str] = []
    used_chars = 0
    for index, hit in enumerate(chunks, start=1):
        source_line = f", url={hit.chunk.source_url}" if hit.chunk.source_url else ""
        block = (
            f"[Source {index}] {hit.chunk.source_title} "
            f"(score={hit.score:.2f}, path={hit.chunk.source_path}{source_line})\n"
            f"{hit.chunk.text.strip()}"
        )
        if used_chars + len(block) > RAG_MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used_chars += len(block)

    return "\n\n".join(parts) if parts else "No external knowledge retrieved."


def generate_mock_feature_plan(dataset_summary: str) -> list[str]:
    """Deterministic fallback used in tests and mock mode."""
    retrieval = retrieve_knowledge(dataset_summary)
    ideas: list[str] = []
    corpus = retrieval.context.lower()

    if "last_dt" in dataset_summary or "time series" in corpus:
        ideas.extend(
            [
                "days_since_last_review from last_dt with NaN -> 9999 to encode listing recency",
                "review_month and review_weekday from last_dt to capture seasonal booking patterns",
            ]
        )
    if "lat" in dataset_summary and "lon" in dataset_summary:
        ideas.append(
            "dist_to_midtown_km from lat/lon using haversine distance because Manhattan proximity drives demand"
        )
    if "location_cluster" in dataset_summary:
        ideas.append(
            "borough_room_interaction from location_cluster x type_house to model borough-specific room premiums"
        )
    if "sum" in dataset_summary:
        ideas.append(
            "log_sum and sum_per_min_day from sum and min_days to stabilize skewed price effects"
        )
    if "amt_reviews" in dataset_summary or "avg_reviews" in dataset_summary:
        ideas.append(
            "has_reviews and reviews_density from amt_reviews/avg_reviews to separate fresh vs established listings"
        )
    if "total_host" in dataset_summary:
        ideas.append(
            "host_portfolio_log from log1p(total_host) because multi-listing hosts often price differently"
        )

    fallback = [
        "fill avg_reviews NaN with 0 because missing usually means no reviews",
        "target encode or label encode location and location_cluster with CV-safe handling",
    ]
    for idea in fallback:
        if idea not in ideas:
            ideas.append(idea)

    return ideas[:8]


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return "lexical"
        return "chroma"
    return backend


def _resolve_embedding_backend(embedding_backend: str):
    if embedding_backend in {"auto", "onnx_mini_lm"}:
        return OnnxMiniLMEmbeddingBackend(cache_dir=RAG_EMBEDDING_CACHE_DIR)
    if embedding_backend == "hash":
        return HashEmbeddingBackend()
    raise ValueError(f"Unsupported embedding backend: {embedding_backend}")


def _hybrid_rerank(
    query: str,
    vector_hits: list[ScoredChunk],
    lexical_hits: list[ScoredChunk],
    *,
    final_k: int,
) -> list[ScoredChunk]:
    """Blend vector similarity with lexical relevance for stable top-k retrieval."""
    merged: dict[str, ScoredChunk] = {}
    combined_scores: dict[str, float] = {}

    for hit in vector_hits:
        merged[hit.chunk.chunk_id] = hit
        combined_scores[hit.chunk.chunk_id] = combined_scores.get(hit.chunk.chunk_id, 0.0) + (
            hit.score * 0.7
        )

    for hit in lexical_hits:
        merged[hit.chunk.chunk_id] = hit
        combined_scores[hit.chunk.chunk_id] = combined_scores.get(hit.chunk.chunk_id, 0.0) + (
            hit.score * 0.9
        )

    for chunk_id, hit in merged.items():
        combined_scores[chunk_id] += _domain_signal_boost(query, hit.chunk.text, hit.chunk.source_title)

    ranked = sorted(
        merged.values(),
        key=lambda hit: combined_scores.get(hit.chunk.chunk_id, 0.0),
        reverse=True,
    )
    return [
        ScoredChunk(chunk=hit.chunk, score=combined_scores[hit.chunk.chunk_id])
        for hit in ranked[:final_k]
    ]


def _domain_signal_boost(query: str, text: str, title: str) -> float:
    query_lower = query.lower()
    haystack = f"{title}\n{text}".lower()
    boost = 0.0

    if _contains_any(query_lower, ("time", "cross validation", "last_dt", "weekday", "month")):
        if _contains_any(haystack, ("time series", "timeseriessplit", "last_dt", "weekday", "month", "recency")):
            boost += 0.45

    if _contains_any(query_lower, ("catboost", "boosting", "categorical")):
        if _contains_any(haystack, ("catboost", "gradient-boosting", "categorical")):
            boost += 0.20

    if _contains_any(query_lower, ("outlier", "validation", "nan")):
        if _contains_any(haystack, ("outlier", "validation", "robust", "nan")):
            boost += 0.15

    return boost


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
