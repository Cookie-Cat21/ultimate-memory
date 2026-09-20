"""Benchmark-agnostic candidate reranking."""
from __future__ import annotations

import re

from .models import SearchResult
from .planner import QueryPlan


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_.-]{3,}", text)
        if t.lower() not in {"what", "where", "when", "which", "with", "from", "that", "this"}
    }


def rerank_candidates(
    query: str,
    results: list[SearchResult],
    plan: QueryPlan,
) -> list[SearchResult]:
    q_tokens = _tokens(query)
    wanted_types = set(plan.memory_types)

    for result in results:
        score = float(result.score)
        text_lower = result.text.lower()
        r_tokens = _tokens(result.text)

        if q_tokens:
            overlap = len(q_tokens & r_tokens) / len(q_tokens)
            score += 0.18 * overlap

        entity_hits = sum(1 for entity in plan.entities if entity.lower() in text_lower)
        score += min(0.18, 0.07 * entity_hits)

        if result.memory_type in wanted_types:
            score += 0.08

        provenance = result.provenance or {}
        claim = provenance.get("claim")
        if isinstance(claim, dict):
            score += 0.04 * float(claim.get("confidence") or 0.0)

        if plan.temporal_mode == "current" and provenance.get("valid_until"):
            score -= 0.35
        elif plan.temporal_mode in {"historical", "as_of"} and provenance.get("valid_until"):
            score += 0.08

        result.provenance["planner_score"] = round(score, 6)
        result.score = max(score, 0.0)

    results.sort(key=lambda item: item.score, reverse=True)
    if results:
        top = max(result.score for result in results)
        if top > 1.0:
            for result in results:
                result.score = result.score / top
    return results
