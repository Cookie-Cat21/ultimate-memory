"""Optional in-process semantic retrieval for local/fallback mode.

This reuses the configured FastEmbed model and keeps embeddings in process memory.
It is deliberately optional so normal unit tests and lightweight installs do not
need to download a model. Enable with ULTIMATE_MEMORY_LOCAL_SEMANTIC=1.
"""
from __future__ import annotations

import math
import os

from .models import AtomicMemory, SearchResult


def enabled() -> bool:
    return os.environ.get("ULTIMATE_MEMORY_LOCAL_SEMANTIC", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln == 0.0 or rn == 0.0:
        return 0.0
    return dot / (ln * rn)


class LocalSemanticIndex:
    def __init__(self, embed) -> None:
        self._embed = embed
        self._vectors: dict[str, tuple[str, list[float]]] = {}

    def _ensure(self, atoms: list[AtomicMemory]) -> None:
        missing = [
            atom for atom in atoms
            if atom.id not in self._vectors or self._vectors[atom.id][0] != atom.content_hash
        ]
        if not missing:
            return
        vectors = self._embed([atom.text for atom in missing])
        for atom, vector in zip(missing, vectors, strict=True):
            self._vectors[atom.id] = (atom.content_hash, vector)

    def search(
        self,
        query: str,
        atoms: list[AtomicMemory],
        *,
        limit: int,
    ) -> list[SearchResult]:
        if not query.strip() or not atoms:
            return []
        self._ensure(atoms)
        query_vector = self._embed([query])[0]

        scored: list[tuple[float, AtomicMemory]] = []
        for atom in atoms:
            cached = self._vectors.get(atom.id)
            if cached is None:
                continue
            score = cosine(query_vector, cached[1])
            scored.append((score, atom))
        scored.sort(key=lambda item: item[0], reverse=True)

        out: list[SearchResult] = []
        for score, atom in scored[:limit]:
            out.append(
                SearchResult(
                    id=atom.id,
                    text=atom.text,
                    title=f"{atom.memory_type.value}: {atom.text[:72]}",
                    source_path=f"atom://{atom.id}",
                    memory_type=atom.memory_type.value,
                    score=max(0.0, min((score + 1.0) / 2.0, 1.0)),
                    provenance={
                        "source": "local-semantic",
                        "semantic_score": score,
                        "salience": atom.salience,
                        "valid_from": atom.valid_from,
                        "valid_until": atom.valid_until,
                        "superseded_by": atom.superseded_by,
                        "entities": atom.entities,
                        "project_path": atom.project_path,
                        "dia_id": atom.metadata.get("dia_id"),
                        "speaker": atom.metadata.get("speaker"),
                        "direct_turn": bool(atom.metadata.get("dia_id")),
                        "claim": atom.metadata.get("claim"),
                    },
                )
            )
        return out
