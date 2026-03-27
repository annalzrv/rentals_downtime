"""Vector retrieval backed by ChromaDB with pluggable embedding backends."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from rentals_agents.rag.knowledge_base import KnowledgeChunk
from rentals_agents.rag.retriever import ScoredChunk, tokenize

try:
    import chromadb
    from chromadb.errors import NotFoundError as ChromaNotFoundError
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - exercised by fallback path only
    chromadb = None
    embedding_functions = None

    class ChromaNotFoundError(Exception):
        """Fallback error type when Chroma is not installed."""


class EmbeddingBackend(Protocol):
    """Embedding interface used by the vector store."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


@dataclass(frozen=True)
class HashEmbeddingBackend:
    """Deterministic offline embedder for tests and no-network environments."""

    dimension: int = 128

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


class OnnxMiniLMEmbeddingBackend:
    """Chroma's built-in ONNX MiniLM embedding backend."""

    def __init__(self, *, cache_dir: str) -> None:
        if embedding_functions is None:  # pragma: no cover - guarded by service
            raise RuntimeError("chromadb is not installed")

        target_dir = Path(cache_dir) / "onnx_models" / "all-MiniLM-L6-v2"
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        embedding_functions.ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(target_dir)
        self._embedder = embedding_functions.DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder(texts)


class ChromaVectorRetriever:
    """Persist and query a Chroma collection for RAG chunks."""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        *,
        persist_dir: str,
        collection_name: str,
        embedding_backend: EmbeddingBackend,
    ) -> None:
        if chromadb is None:  # pragma: no cover - guarded by service
            raise RuntimeError("chromadb is not installed")

        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._embedder = embedding_backend
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name
        self._collection = self._rebuild_collection(chunks)

    def search(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        if not query.strip():
            return []

        query_embedding = self._embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas"],
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        items: list[ScoredChunk] = []
        for chunk_id, distance in zip(ids, distances):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            score = 1.0 / (1.0 + float(distance if distance is not None else math.inf))
            items.append(ScoredChunk(chunk=chunk, score=score))
        return items

    def _rebuild_collection(self, chunks: list[KnowledgeChunk]):
        try:
            self._client.delete_collection(self._collection_name)
        except ChromaNotFoundError:
            pass
        except ValueError:
            pass

        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"source": "rentals_agents_rag"},
        )
        if not chunks:
            return collection

        embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_id": chunk.source_id,
                    "source_title": chunk.source_title,
                    "source_path": chunk.source_path,
                    "source_url": chunk.source_url or "",
                }
                for chunk in chunks
            ],
        )
        return collection
