"""Compact retrieved evidence into a diverse, token-efficient context packet."""
from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z0-9_-]{2,}")


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD_RE.finditer(text)}


def _similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _source_bucket(item: dict) -> str:
    provenance = item.get("provenance") or {}
    session_id = provenance.get("session_id")
    if session_id:
        return f"session:{session_id}"
    source = str(provenance.get("source") or "")
    source_path = str(item.get("source_path") or "")
    return source_path or source or str(item.get("id") or "unknown")


def compile_context_packet(
    contexts: list[dict],
    *,
    max_chars: int = 12000,
    max_items: int = 24,
    near_duplicate_threshold: float = 0.84,
    max_per_source: int = 6,
) -> list[dict]:
    """Select high-ranked, diverse evidence under a deterministic size budget.

    The input order is treated as ranking order. Near-duplicates and repeated
    copies of the same session are suppressed, but small direct-turn evidence
    can coexist with a richer conversation window when they are materially
    different.
    """
    if not contexts or max_chars <= 0 or max_items <= 0:
        return []

    selected: list[dict] = []
    selected_texts: list[str] = []
    source_counts: Counter[str] = Counter()
    used_chars = 0

    for item in contexts:
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        bucket = _source_bucket(item)
        if source_counts[bucket] >= max_per_source:
            continue

        if any(
            _similarity(text, existing) >= near_duplicate_threshold
            for existing in selected_texts
        ):
            continue

        remaining = max_chars - used_chars
        if remaining <= 0:
            break

        # Keep evidence boundaries intact whenever possible. Only truncate one
        # oversized first item rather than filling the packet with fragments.
        if len(text) > remaining:
            if selected:
                continue
            clipped = text[:remaining].rsplit(" ", 1)[0].rstrip()
            if not clipped:
                continue
            item = dict(item)
            item["text"] = clipped
            text = clipped

        selected.append(item)
        selected_texts.append(text)
        source_counts[bucket] += 1
        used_chars += len(text) + 2
        if len(selected) >= max_items:
            break

    return selected
