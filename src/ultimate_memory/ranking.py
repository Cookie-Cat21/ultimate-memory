"""Benchmark-agnostic candidate reranking."""
from __future__ import annotations

import re
from datetime import datetime

from .models import SearchResult
from .planner import QueryPlan


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_.-]{3,}", text)
        if t.lower() not in {"what", "where", "when", "which", "with", "from", "that", "this"}
    }


def _provenance_value(provenance: dict, key: str):
    """Read a provenance field across direct, metadata, and vector payload forms."""
    value = provenance.get(key)
    if value not in (None, "", [], {}):
        return value
    metadata = provenance.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    payload = provenance.get("payload")
    if isinstance(payload, dict):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def filter_entity_scoped_results(
    results: list[dict],
    entities: list[str],
    *,
    min_matches: int = 2,
) -> list[dict]:
    """Prefer evidence *attributed to* entities named in the query.

    Conversational memory frequently contains windows with two speakers. A name
    appearing in the question half of a pair is weaker evidence than a direct
    turn or the answer half of that pair. This gate therefore separates strong
    attribution from mere mentions before balancing multiple named entities.
    """
    entity_keys = [entity.strip().casefold() for entity in entities if entity.strip()]
    if not entity_keys or not results:
        return list(results)

    known_speakers: set[str] = set()
    for item in results:
        provenance = item.get("provenance") or {}
        for key in ("speaker", "answer_speaker"):
            speaker = str(_provenance_value(provenance, key) or "").strip().casefold()
            if speaker:
                known_speakers.add(speaker)

    speaker_keys = [key for key in entity_keys if key in known_speakers]
    scope_keys = speaker_keys or entity_keys

    strong: list[dict] = []
    weak: list[dict] = []
    strong_buckets: dict[str, list[dict]] = {key: [] for key in scope_keys}
    weak_buckets: dict[str, list[dict]] = {key: [] for key in scope_keys}

    for item in results:
        text = str(item.get("text") or "").casefold()
        provenance = item.get("provenance") or {}
        prov_entities = {
            str(entity).strip().casefold()
            for entity in (_provenance_value(provenance, "entities") or [])
            if str(entity).strip()
        }
        speaker = str(_provenance_value(provenance, "speaker") or "").strip().casefold()
        answer_speaker = str(
            _provenance_value(provenance, "answer_speaker") or ""
        ).strip().casefold()
        question_speaker = str(
            _provenance_value(provenance, "question_speaker") or ""
        ).strip().casefold()

        strong_entities: list[str] = []
        weak_entities: list[str] = []
        for key in scope_keys:
            if key == speaker or key == answer_speaker or key in prov_entities:
                strong_entities.append(key)
            elif key in text or key == question_speaker:
                weak_entities.append(key)

        if strong_entities:
            strong.append(item)
            for key in strong_entities:
                strong_buckets[key].append(item)
        elif weak_entities:
            weak.append(item)
            for key in weak_entities:
                weak_buckets[key].append(item)

    # Strongly attributed evidence is ordered first, but weaker mention /
    # question-side evidence is retained as recall support. Hard-dropping it can
    # lose conversational premises whose answer lives in an adjacent turn.
    matched = list(strong)
    seen = {
        str(item.get("id") or item.get("source_path") or id(item))
        for item in matched
    }
    for item in weak:
        item_key = str(item.get("id") or item.get("source_path") or id(item))
        if item_key not in seen:
            matched.append(item)
            seen.add(item_key)

    buckets = {
        key: [*strong_buckets[key], *weak_buckets[key]]
        for key in scope_keys
    }

    if len(matched) < min_matches:
        return list(results)

    if len(scope_keys) == 1 or not all(buckets[key] for key in scope_keys):
        return matched

    # Round-robin per-entity evidence so one person's larger history cannot
    # crowd another named person out of the context packet.
    balanced: list[dict] = []
    seen: set[str] = set()
    max_len = max(len(bucket) for bucket in buckets.values())
    for index in range(max_len):
        for key in scope_keys:
            bucket = buckets[key]
            if index >= len(bucket):
                continue
            item = bucket[index]
            item_key = str(item.get("id") or item.get("source_path") or id(item))
            if item_key in seen:
                continue
            seen.add(item_key)
            balanced.append(item)

    for item in matched:
        item_key = str(item.get("id") or item.get("source_path") or id(item))
        if item_key in seen:
            continue
        seen.add(item_key)
        balanced.append(item)
    return balanced



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



def _event_timestamp(result: SearchResult) -> float | None:
    provenance = result.provenance or {}
    raw = provenance.get("event_time") or provenance.get("created_at")
    if not raw:
        metadata = provenance.get("metadata")
        if isinstance(metadata, dict):
            raw = metadata.get("event_time") or metadata.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def rerank_candidates(
    query: str,
    results: list[SearchResult],
    plan: QueryPlan,
) -> list[SearchResult]:
    q_tokens = _tokens(query)
    wanted_types = set(plan.memory_types)
    query_quantities = _duration_quantities(query)
    timestamps = [ts for result in results if (ts := _event_timestamp(result)) is not None]
    newest_timestamp = max(timestamps) if timestamps else None
    oldest_timestamp = min(timestamps) if timestamps else None

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

        if plan.temporal_mode == "current":
            if provenance.get("valid_until"):
                score -= 0.35
            event_ts = _event_timestamp(result)
            if (
                event_ts is not None
                and newest_timestamp is not None
                and oldest_timestamp is not None
                and newest_timestamp > oldest_timestamp
            ):
                recency = (event_ts - oldest_timestamp) / (newest_timestamp - oldest_timestamp)
                score += 0.22 * recency
                provenance["recency_score"] = round(recency, 6)
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
