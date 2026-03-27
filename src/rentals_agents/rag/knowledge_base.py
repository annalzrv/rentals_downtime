"""Utilities for loading and chunking the local RAG knowledge base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class SourceDocument:
    """A source file stored in the local knowledge base."""

    source_id: str
    path: Path
    title: str
    text: str
    source_url: str | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class KnowledgeChunk:
    """A chunk of text that can be ranked during retrieval."""

    chunk_id: str
    source_id: str
    source_title: str
    source_path: str
    text: str
    source_url: str | None = None


def load_source_documents(knowledge_base_dir: str | Path) -> list[SourceDocument]:
    """Load `.md` and `.txt` files from the knowledge base directory."""
    base_dir = Path(knowledge_base_dir)
    if not base_dir.exists():
        return []

    manifest = load_source_manifest(base_dir)
    documents: list[SourceDocument] = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        source_id = path.relative_to(base_dir).as_posix()
        metadata = manifest.get(source_id, {})
        title = metadata.get("title") or _extract_title(path, text)
        documents.append(
            SourceDocument(
                source_id=source_id,
                path=path,
                title=title,
                text=text,
                source_url=metadata.get("url"),
                source_kind=metadata.get("kind"),
            )
        )
    return documents


def build_knowledge_chunks(
    documents: list[SourceDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    """Split loaded documents into overlapping text chunks."""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for index, chunk_text in enumerate(
            split_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.source_id}#chunk-{index}",
                    source_id=document.source_id,
                    source_title=document.title,
                    source_path=str(document.path),
                    text=chunk_text,
                    source_url=document.source_url,
                )
            )
    return chunks


def load_source_manifest(knowledge_base_dir: str | Path) -> dict[str, dict[str, str]]:
    """Load optional source metadata stored in `sources.json`."""
    manifest_path = Path(knowledge_base_dir) / "sources.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk text by paragraph while keeping mild overlap for retrieval continuity."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            chunks.extend(_hard_split(paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            current = ""

    if current:
        chunks.append(current)

    return chunks


def _hard_split(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fallback splitting for long paragraphs."""
    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        start += step
    return [chunk for chunk in chunks if chunk]


def _extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()
