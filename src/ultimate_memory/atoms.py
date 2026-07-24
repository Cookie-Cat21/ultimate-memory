"""Atomic bi-temporal memory engine.

This is the layer that separates Ultimate Memory from typical RAG notebooks:

- Memories are *atoms* (typed, addressable facts/decisions/preferences/procedures)
- Each atom is bi-temporal: ``created_at`` + ``valid_from`` / ``valid_until``
- Contradictions auto-supersede older atoms instead of leaving stale truth around
- Salience blends type importance, recency decay, and reinforcement on access
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from typing import Iterable

from .models import AtomicMemory, MemoryType, ReflectionPayload, safe_slug

# Negation / replacement cues that often mean "this replaces prior belief".
_SUPERSESSION_CUES = (
    "instead of",
    "no longer",
    "replaces",
    "replaced",
    "switched to",
    "switching to",
    "moved to",
    "migrated to",
    "don't",
    "do not",
    "never",
    "stop using",
    "not using",
    "rather than",
    "as opposed to",
)

_TYPE_IMPORTANCE: dict[str, float] = {
    MemoryType.PREFERENCE.value: 1.0,
    MemoryType.PROCEDURE.value: 0.95,
    MemoryType.DECISION.value: 0.9,
    MemoryType.FACT.value: 0.75,
    MemoryType.NOTE.value: 0.55,
    MemoryType.LOG.value: 0.35,
}

# Half-life in days for salience decay (preferences last longer than facts).
_TYPE_HALF_LIFE_DAYS: dict[str, float] = {
    MemoryType.PREFERENCE.value: 90.0,
    MemoryType.PROCEDURE.value: 60.0,
    MemoryType.DECISION.value: 45.0,
    MemoryType.FACT.value: 30.0,
    MemoryType.NOTE.value: 21.0,
    MemoryType.LOG.value: 14.0,
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./+-]{1,}", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "it",
        "we",
        "i",
        "you",
        "they",
        "from",
        "as",
        "at",
        "by",
        "use",
        "using",
        "used",
        "should",
        "will",
        "can",
        "our",
        "their",
        "going",
        "forward",
        "today",
    }
)


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).lower().strip("._/-+")
        if token and token not in _STOPWORDS and len(token) > 2:
            tokens.add(token)
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


_TEMPORAL_QUERY_CUES = (
    "before",
    "previously",
    "used to",
    "back when",
    "earlier",
    "formerly",
    "old ",
    "prior to",
    "at the time",
)


def is_temporal_query(query: str) -> bool:
    """True when the query likely asks about past / superseded knowledge."""
    lower = query.lower()
    return any(cue in lower for cue in _TEMPORAL_QUERY_CUES)


def atom_valid_at(atom: AtomicMemory, as_of: str) -> bool:
    """Return whether *atom* was valid at the given ISO timestamp."""
    as_of_dt = parse_iso(as_of)
    if as_of_dt is None:
        return False
    valid_from = parse_iso(atom.valid_from)
    if valid_from is not None:
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=UTC)
        if valid_from > as_of_dt:
            return False
    valid_until = parse_iso(atom.valid_until)
    if valid_until is not None:
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=UTC)
        if valid_until <= as_of_dt:
            return False
    return True


def days_since(ts: str | None, *, now: datetime | None = None) -> float:
    when = parse_iso(ts)
    if when is None:
        return 0.0
    current = now or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((current - when).total_seconds() / 86400.0, 0.0)


def compute_salience(
    memory_type: str,
    *,
    created_at: str,
    last_accessed: str | None,
    access_count: int,
    valid_until: str | None,
    importance: float | None = None,
    now: datetime | None = None,
) -> float:
    """Salience in [0, 1]: type importance × decay × reinforcement × validity."""
    if valid_until:
        return 0.0
    base = importance if importance is not None else _TYPE_IMPORTANCE.get(memory_type, 0.5)
    half_life = _TYPE_HALF_LIFE_DAYS.get(memory_type, 30.0)
    age = days_since(last_accessed or created_at, now=now)
    decay = math.exp(-math.log(2) * age / half_life)
    reinforcement = 1.0 + 0.15 * math.log1p(max(access_count, 0))
    # Leave headroom so reinforcement can still raise highly important types.
    return max(0.0, min(0.82 * base * decay * reinforcement, 1.0))


def has_supersession_cue(text: str) -> bool:
    lower = text.lower()
    return any(cue in lower for cue in _SUPERSESSION_CUES)


# Role/topic cues for location and job updates (LoCoMo-style life facts).
_LOCATION_CUES = (
    "lives in",
    "live in",
    "living in",
    "moved to",
    "moving to",
    "based in",
    "located in",
    "relocated to",
)
_JOB_CUES = (
    "works as",
    "work as",
    "working as",
    "employed as",
    "job is",
    "i'm a",
    "i am a",
    "im a",
)
# Cue verbs to skip when extracting role *values* (city, job title, etc.).
_ROLE_CUE_WORDS = frozenset(
    {
        "lives",
        "live",
        "living",
        "moved",
        "moving",
        "based",
        "located",
        "relocated",
        "works",
        "work",
        "working",
        "employed",
        "job",
        "staying",
    }
)

# Stop value extraction at supersession tails or secondary prepositional phrases.
_ROLE_VALUE_BOUNDARY = re.compile(
    r"\b(?:instead of|rather than|as opposed to|no longer|not|but|from|near|last)\b|;|,|\s+at\s+",
    re.IGNORECASE,
)


def _has_role_cue(text: str, cues: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(cue in lower for cue in cues)


def _role_value_fragment(tail: str) -> str:
    """Keep only the primary role value before supersession or 'at …' tails."""
    parts = _ROLE_VALUE_BOUNDARY.split(tail.strip(), maxsplit=1)
    return parts[0].strip()


def _role_value_tokens(text: str, cues: tuple[str, ...]) -> set[str]:
    """Tokens after a role cue phrase (e.g. city after 'lives in', title after 'works as')."""
    lower = text.lower()
    values: set[str] = set()
    for cue in cues:
        idx = lower.find(cue)
        if idx < 0:
            continue
        tail = text[idx + len(cue) :]
        tail = re.sub(r"^(?:a|an|the)\s+", "", tail.strip(), flags=re.IGNORECASE)
        fragment = _role_value_fragment(tail)
        count = 0
        for match in _TOKEN_RE.finditer(fragment):
            token = match.group(0).lower().strip("._/-+")
            if token and token not in _STOPWORDS and token not in _ROLE_CUE_WORDS and len(token) > 2:
                values.add(token)
                count += 1
                if count >= 4:
                    break
    return values


def _same_subject(new_text: str, old_text: str) -> bool:
    """True when both texts appear to describe the same person."""
    if re.match(r"^I\b", new_text.strip()) and re.match(r"^I\b", old_text.strip()):
        return True
    new_match = re.match(r"^([A-Z][a-z]+)\b", new_text.strip())
    old_match = re.match(r"^([A-Z][a-z]+)\b", old_text.strip())
    if new_match and old_match:
        return new_match.group(1) == old_match.group(1)
    return False


def contradiction_lookup_hints(text: str, *, entities: list[str] | None = None) -> list[str]:
    """Search hints for finding prior atoms that may contradict *text*."""
    hints: list[str] = []
    if entities:
        hints.extend(e for e in entities if e)
    subject = re.match(r"^([A-Z][a-z]+)\b", text.strip())
    if subject and subject.group(1) not in {"We", "The", "This", "That"}:
        hints.append(subject.group(1))
    if re.match(r"^I\b", text.strip()):
        hints.append("I")
    lower = text.lower()
    for topic in ("editor", "theme", "mode", "caching", "cache"):
        if topic in lower:
            hints.append(topic)
    return list(dict.fromkeys(hints))


def role_conflict_boost(new_text: str, old_text: str) -> float:
    """Extra contradiction score when location/job role facts change."""
    boost = 0.0
    new_lower = new_text.lower()
    old_lower = old_text.lower()
    explicit_replace = has_supersession_cue(new_text)

    if _has_role_cue(new_text, _LOCATION_CUES) and _has_role_cue(old_text, _LOCATION_CUES):
        new_vals = _role_value_tokens(new_text, _LOCATION_CUES)
        old_vals = _role_value_tokens(old_text, _LOCATION_CUES)
        if new_vals and old_vals:
            if new_vals == old_vals:
                boost -= 0.15
            elif jaccard(new_vals, old_vals) < 0.45:
                boost += 0.38
            elif jaccard(new_vals, old_vals) >= 0.55 and not explicit_replace:
                boost -= 0.12
        if "moved to" in new_lower and ("lives in" in old_lower or "live in" in old_lower):
            if new_vals and old_vals and new_vals != old_vals:
                boost += 0.28
        if "moved to" in new_lower and "moved to" in old_lower and new_vals != old_vals:
            boost += 0.18

    if _has_role_cue(new_text, _JOB_CUES) and _has_role_cue(old_text, _JOB_CUES):
        new_vals = _role_value_tokens(new_text, _JOB_CUES)
        old_vals = _role_value_tokens(old_text, _JOB_CUES)
        if new_vals and old_vals and jaccard(new_vals, old_vals) < 0.45:
            boost += 0.36
        elif new_vals and old_vals and new_vals != old_vals and explicit_replace:
            boost += 0.28
        shared_roles = (tokenize(new_text) & tokenize(old_text)) & _ROLE_CUE_WORDS
        if shared_roles and new_vals and old_vals and jaccard(new_vals, old_vals) < 0.5:
            boost += 0.12

    if explicit_replace and boost > 0:
        boost += 0.08
    if _same_subject(new_text, old_text) and boost > 0:
        boost += 0.06

    return max(boost, 0.0)


def contradiction_score(new_text: str, old_text: str) -> float:
    """Heuristic contradiction score in [0, 1].

    High lexical overlap plus polarity/supersession cues → likely contradiction.
    Exact/near-duplicate text scores high for consolidation, not supersession —
    callers should treat score >= 0.85 with near-identical text as a duplicate.
    """
    new_tokens = tokenize(new_text)
    old_tokens = tokenize(old_text)
    overlap = jaccard(new_tokens, old_tokens)
    role_boost = role_conflict_boost(new_text, old_text)
    same_subject = _same_subject(new_text, old_text)
    min_role = 0.30 if same_subject and has_supersession_cue(new_text) else 0.35
    if overlap < 0.28 and role_boost < min_role:
        return 0.0

    score = overlap if overlap >= 0.28 else 0.0
    if has_supersession_cue(new_text):
        score += 0.18
    if same_subject and has_supersession_cue(new_text):
        score += 0.08
    if content_hash(new_text) == content_hash(old_text):
        return 1.0

    # Opposite polarity on shared topic words (prefer X vs never X / don't X)
    new_neg = bool(re.search(r"\b(never|don't|do not|avoid|stop)\b", new_text.lower()))
    old_neg = bool(re.search(r"\b(never|don't|do not|avoid|stop)\b", old_text.lower()))
    new_pos = bool(re.search(r"\b(always|prefer|use|should)\b", new_text.lower()))
    old_pos = bool(re.search(r"\b(always|prefer|use|should)\b", old_text.lower()))
    if (new_neg and old_pos) or (new_pos and old_neg):
        score += 0.22

    if overlap < 0.28:
        if role_boost < min_role:
            return 0.0
        score = max(score, role_boost)
    else:
        score += role_boost

    return min(score, 1.0)


def atoms_from_reflection(
    payload: ReflectionPayload,
    *,
    project_path: str | None = None,
    entities: list[str] | None = None,
    created_at: str | None = None,
    event_time: str | None = None,
) -> list[AtomicMemory]:
    """Expand a reflection payload into typed atomic memories."""
    stamp = created_at or now_iso()
    event_stamp = event_time or stamp
    entity_list = list(entities or [])
    buckets: list[tuple[MemoryType, list[str]]] = [
        (MemoryType.FACT, payload.facts),
        (MemoryType.PREFERENCE, payload.preferences),
        (MemoryType.DECISION, payload.decisions),
        (MemoryType.PROCEDURE, payload.procedures),
    ]
    atoms: list[AtomicMemory] = []
    for memory_type, items in buckets:
        for item in items:
            text = item.strip()
            if len(text) < 12:
                continue
            atom_id = f"atom:{memory_type.value}:{safe_slug(text)[:48]}:{content_hash(text)[:8]}"
            atoms.append(
                AtomicMemory(
                    id=atom_id,
                    text=text,
                    memory_type=memory_type,
                    project_path=project_path,
                    entities=entity_list,
                    source_refs=list(payload.source_refs),
                    created_at=stamp,
                    valid_from=stamp,
                    event_time=event_stamp,
                    importance=_TYPE_IMPORTANCE.get(memory_type.value, 0.5),
                    content_hash=content_hash(text),
                    metadata={"from_summary": payload.summary[:160]},
                )
            )
    return atoms


def blend_scores(
    rrf_score: float,
    salience: float,
    *,
    rrf_weight: float = 0.72,
    salience_weight: float = 0.28,
) -> float:
    total = rrf_weight + salience_weight
    return ((rrf_weight * rrf_score) + (salience_weight * salience)) / total


def group_near_duplicates(atoms: Iterable[AtomicMemory], threshold: float = 0.72) -> list[list[AtomicMemory]]:
    """Group active atoms that are near-duplicates for consolidation."""
    active = [a for a in atoms if a.is_active]
    groups: list[list[AtomicMemory]] = []
    used: set[str] = set()
    for i, atom in enumerate(active):
        if atom.id in used:
            continue
        group = [atom]
        used.add(atom.id)
        tokens_i = tokenize(atom.text)
        for other in active[i + 1 :]:
            if other.id in used:
                continue
            if other.memory_type != atom.memory_type:
                continue
            if jaccard(tokens_i, tokenize(other.text)) >= threshold:
                group.append(other)
                used.add(other.id)
        if len(group) > 1:
            groups.append(group)
    return groups
