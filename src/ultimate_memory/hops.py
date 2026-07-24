"""Multi-hop memory retrieval helpers.

After an initial search, extract bridge entities (e.g. Fiona from
"Elena's sister is named Fiona") and run capped follow-up searches.
"""

from __future__ import annotations

import re

MAX_HOP_SEARCHES = 3

_CAP_TOKEN_RE = re.compile(r"\b([A-Z][a-z]+(?:'s)?)\b")
_POSSESSIVE_RE = re.compile(r"^(.+)'s$")

# Capitalized tokens that are usually not person/entity names.
_SKIP_CAPITALIZED = frozenset(
    {
        "A",
        "An",
        "The",
        "What",
        "When",
        "Where",
        "Who",
        "Whom",
        "Which",
        "How",
        "Why",
        "Does",
        "Do",
        "Did",
        "Is",
        "Are",
        "Was",
        "Were",
        "Has",
        "Have",
        "Had",
        "Can",
        "Could",
        "Would",
        "Should",
        "Will",
        "May",
        "Might",
        "Before",
        "After",
        "During",
        "Instead",
        "Named",
        "Called",
        "Works",
        "Work",
        "Lives",
        "Live",
        "Moved",
        "Move",
        "Delta",
        "Airlines",
    }
)


def normalize_entity(name: str) -> str:
    """Strip possessive suffix and normalize for comparison."""
    cleaned = name.strip()
    match = _POSSESSIVE_RE.match(cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned.lower()


def extract_capitalized_entities(text: str) -> list[str]:
    """Return capitalized tokens that look like person/entity names."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _CAP_TOKEN_RE.finditer(text):
        token = match.group(1)
        base = token
        poss = _POSSESSIVE_RE.match(token)
        if poss:
            base = poss.group(1)
        if base in _SKIP_CAPITALIZED or len(base) < 2:
            continue
        key = base.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(base)
    return found


def extract_hop_entities(
    question: str,
    results: list[dict],
    *,
    limit: int = 8,
) -> list[str]:
    """Collect entity names from the question, top results, and atom metadata."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        base = name.strip()
        poss = _POSSESSIVE_RE.match(base)
        if poss:
            base = poss.group(1)
        if not base or base in _SKIP_CAPITALIZED or len(base) < 2:
            return
        key = base.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(base)

    for item in results:
        provenance = item.get("provenance") or {}
        for ent in provenance.get("entities") or []:
            if isinstance(ent, str):
                add(ent)

    atom_results = [
        item
        for item in results
        if (item.get("provenance") or {}).get("source") == "atomic-memory"
        or str(item.get("id", "")).startswith("atom:")
    ]
    for item in atom_results:
        for field in ("text", "title"):
            value = item.get(field) or ""
            for ent in extract_capitalized_entities(value):
                add(ent)
        text = item.get("text") or ""
        for match in re.finditer(
            r"\b(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            text,
        ):
            add(match.group(1))

    for ent in extract_capitalized_entities(question):
        add(ent)

    return ordered[:limit]


def build_hop_queries(
    question: str,
    entities: list[str],
    *,
    max_queries: int = 2,
) -> list[str]:
    """Build follow-up search queries for bridge entities discovered in hop 1."""
    if not entities:
        return []

    q_lower = question.lower()
    question_entity_keys = {normalize_entity(ent) for ent in extract_capitalized_entities(question)}

    # Prefer bridge entities not already named in the question (e.g. Fiona, not Elena).
    bridge = [ent for ent in entities if normalize_entity(ent) not in question_entity_keys]
    candidates = bridge or list(entities)

    queries: list[str] = []
    for entity in candidates:
        if len(queries) >= max_queries:
            break
        if any(
            cue in q_lower
            for cue in ("work", "job", "do for a living", "occupation", "career", "employer")
        ):
            queries.append(f"{entity} work job")
        elif any(cue in q_lower for cue in ("where", "live", "located", "city", "address")):
            queries.append(f"{entity} lives location")
        elif any(cue in q_lower for cue in ("when", "move", "start", "begin", "date")):
            queries.append(f"{entity} when date")
        elif "who" in q_lower or "name" in q_lower:
            queries.append(f"{entity} named called")
        else:
            # Generic: anchor on the entity plus salient question terms.
            terms = [t for t in re.findall(r"[a-z]{3,}", q_lower) if t not in {"what", "does", "the"}]
            tail = " ".join(terms[:3]) if terms else "facts"
            queries.append(f"{entity} {tail}")

    return queries[:max_queries]


def merge_contexts(
    initial: list[str] | list[dict],
    *extra_lists: list[str] | list[dict],
) -> list[str] | list[dict]:
    """Merge context strings or search-result dicts, preserving order."""
    merged: list[str] | list[dict] = list(initial)
    seen: set[str] = set()

    def key_for(item: str | dict) -> str | None:
        if isinstance(item, str):
            text = item.strip()
        else:
            text = (item.get("text") or "").strip()
        if not text:
            return None
        return text.lower()

    for item in initial:
        item_key = key_for(item)
        if item_key:
            seen.add(item_key)

    for extra in extra_lists:
        for item in extra:
            item_key = key_for(item)
            if not item_key or item_key in seen:
                continue
            seen.add(item_key)
            merged.append(item)
    return merged
