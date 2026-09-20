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


def filter_entity_scoped_results(
    results: list[dict],
    entities: list[str],
    *,
    min_matches: int = 2,
) -> list[dict]:
    """Prefer evidence explicitly attached to the entities named in the query.

    This is intentionally a post-retrieval gate rather than a storage filter:
    global memories remain available, and multi-hop callers can simply skip this
    helper when bridge-entity traversal is required.
    """
    entity_keys = [entity.strip().casefold() for entity in entities if entity.strip()]
    if not entity_keys or not results:
        return list(results)

    matched: list[dict] = []
    for item in results:
        text = str(item.get("text") or "").casefold()
        provenance = item.get("provenance") or {}
        prov_entities = {
            str(entity).strip().casefold()
            for entity in provenance.get("entities") or []
            if str(entity).strip()
        }
        speaker = str(provenance.get("speaker") or "").strip().casefold()

        if any(
            key in text
            or key == speaker
            or key in prov_entities
            for key in entity_keys
        ):
            matched.append(item)

    # Never collapse a query to an unusably tiny evidence pool.
    return matched if len(matched) >= min_matches else list(results)



_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}


def _duration_quantities(text: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    pattern = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+(years?|months?|weeks?|days?|hours?)\b",
        re.I,
    )
    for match in pattern.finditer(text):
        raw = match.group(1).lower()
        number = _NUMBER_WORDS.get(raw, raw)
        unit = match.group(2).lower().rstrip("s")
        out.add((number, unit))
    return out


def rerank_candidates(
    query: str,
    results: list[SearchResult],
    plan: QueryPlan,
) -> list[SearchResult]:
    q_tokens = _tokens(query)
    wanted_types = set(plan.memory_types)
    query_quantities = _duration_quantities(query)

    for result in results:
        score = float(result.score)
        text_lower = result.text.lower()
        r_tokens = _tokens(result.text)

        if q_tokens:
            overlap = len(q_tokens & r_tokens) / len(q_tokens)
            score += 0.18 * overlap

        entity_hits = sum(1 for entity in plan.entities if entity.lower() in text_lower)
        score += min(0.18, 0.07 * entity_hits)

        if query_quantities:
            result_quantities = _duration_quantities(result.text)
            if result_quantities:
                if query_quantities & result_quantities:
                    score += 0.18
                elif any(
                    q_unit == r_unit
                    for _, q_unit in query_quantities
                    for _, r_unit in result_quantities
                ):
                    score -= 0.18

        if result.memory_type in wanted_types:
            score += 0.08

        provenance = result.provenance or {}
        claim = provenance.get("claim")
        if isinstance(claim, dict):
            score += 0.04 * float(claim.get("confidence") or 0.0)

        # Direct dialogue/tool observations are primary evidence. Derived reflections
        # remain useful, but should not outrank an equally relevant original turn.
        if provenance.get("direct_turn") or provenance.get("dia_id"):
            direct_overlap = len(q_tokens & r_tokens) / len(q_tokens) if q_tokens else 0.0
            score += 0.12 + 0.18 * direct_overlap

        if provenance.get("compiled"):
            confidence = float(provenance.get("compiler_confidence") or 0.0)
            score += min(0.10, max(0.0, confidence) * 0.10)

        if "auto-extracted from" in text_lower:
            score -= 0.08

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
