"""A lightweight lexical retriever for the local RAG knowledge base."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from rentals_agents.rag.knowledge_base import KnowledgeChunk


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk paired with its relevance score."""

    chunk: KnowledgeChunk
    score: float


class LexicalRetriever:
    """Rank chunks using token overlap with a simple IDF-style boost."""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = chunks
        self._doc_freq = self._build_doc_freq(chunks)

    def search(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored: list[ScoredChunk] = []
        for chunk in self._chunks:
            score = self._score_chunk(query_tokens, chunk)
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _score_chunk(self, query_tokens: list[str], chunk: KnowledgeChunk) -> float:
        chunk_tokens = tokenize(chunk.text)
        if not chunk_tokens:
            return 0.0

        token_counts: dict[str, int] = {}
        for token in chunk_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        unique_tokens = set(chunk_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in unique_tokens:
                continue
            score += (1 + math.log1p(token_counts[token])) * self._idf(token)

        title_tokens = set(tokenize(chunk.source_title))
        score += 1.2 * len(title_tokens.intersection(query_tokens))
        return score

    def _idf(self, token: str) -> float:
        df = self._doc_freq.get(token, 0)
        return math.log((1 + len(self._chunks)) / (1 + df)) + 1.0

    @staticmethod
    def _build_doc_freq(chunks: Iterable[KnowledgeChunk]) -> dict[str, int]:
        doc_freq: dict[str, int] = {}
        for chunk in chunks:
            for token in set(tokenize(chunk.text)):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        return doc_freq


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase tokens suitable for lexical matching."""
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]
