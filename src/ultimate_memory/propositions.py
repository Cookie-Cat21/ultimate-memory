"""Benchmark-agnostic proposition extraction from dialogue turns.

The goal is not semantic rewriting. We split a turn into compact factual units and
resolve first-person references to the speaker so entity-centric retrieval can
connect facts across sessions without requiring an LLM at ingestion time.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"\s*;\s*")
_SPACE = re.compile(r"\s+")


def ground_speaker_reference(speaker: str, text: str) -> str:
    """Replace first-person references with the explicit speaker name.

    This is intentionally conservative: it only rewrites first-person pronouns,
    leaving second/third-person references untouched.
    """
    value = text.strip()
    if not value:
        return value

    replacements = (
        (re.compile(r"\bI\s+am\b", re.I), f"{speaker} is"),
        (re.compile(r"\bI'm\b", re.I), f"{speaker} is"),
        (re.compile(r"\bI\s+have\b", re.I), f"{speaker} has"),
        (re.compile(r"\bI've\b", re.I), f"{speaker} has"),
        (re.compile(r"\bI'll\b", re.I), f"{speaker} will"),
        (re.compile(r"\bI'd\b", re.I), f"{speaker} would"),
        (re.compile(r"\bmyself\b", re.I), speaker),
        (re.compile(r"\bmine\b", re.I), f"{speaker}'s"),
        (re.compile(r"\bmy\b", re.I), f"{speaker}'s"),
        (re.compile(r"\bme\b", re.I), speaker),
        (re.compile(r"\bI\b", re.I), speaker),
    )
    for pattern, replacement in replacements:
        value = pattern.sub(replacement, value)

    return _SPACE.sub(" ", value).strip()


def turn_propositions(
    speaker: str,
    utterance: str,
    *,
    min_chars: int = 12,
    max_chars: int = 360,
    max_items: int = 8,
) -> list[str]:
    """Return compact speaker-grounded propositions from one dialogue turn."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(utterance.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        clauses = _CLAUSE_SPLIT.split(sentence)
        pieces.extend(clause.strip() for clause in clauses if clause.strip())

    result: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        grounded = ground_speaker_reference(speaker, piece)
        grounded = grounded.strip(" ")
        if not grounded:
            continue
        if len(grounded) < min_chars or len(grounded) > max_chars:
            continue
        key = re.sub(r"\W+", " ", grounded).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(grounded)
        if len(result) >= max_items:
            break
    return result
