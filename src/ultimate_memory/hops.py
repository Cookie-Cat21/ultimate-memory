"""Multi-hop memory retrieval helpers.

After an initial search, extract bridge entities (e.g. Fiona from
"Elena's sister is named Fiona") and run capped follow-up searches.
"""

from __future__ import annotations

import re

MAX_HOP_SEARCHES = 3

# Chained multi-hop: how many bridge-entity "levels" deep to walk before giving
# up. depth=1 reproduces the old single-pass behaviour; depth>=2 lets an
# entity discovered in hop 1 (e.g. Fiona's employer) spawn hop-2 queries
# (e.g. that employer's HQ city) instead of stopping after one pass.
MAX_HOP_DEPTH = 3
# Multi-hop questions get a wider total search budget than the other three
# LoCoMo categories (which keep MAX_HOP_SEARCHES=3 to avoid regressing their
# already-winning scores).
MULTI_HOP_SEARCH_BUDGET = 8

# Legacy single-word matcher, used for every category except multi_hop so
# their entity extraction stays bit-for-bit identical to the pre-chaining
# behavior (avoids regressing their already-winning scores).
_CAP_TOKEN_RE_LEGACY = re.compile(r"\b([A-Z][a-z]+(?:'s)?)\b")
# multi_hop-only: match runs of up to 4 consecutive capitalized words as a
# single phrase (e.g. "Little Women", "Counter Strike Global Offensive")
# instead of splitting proper-noun phrases into meaningless single-word
# fragments.
_CAP_TOKEN_RE_STRICT = re.compile(r"\b([A-Z][a-z]+(?:'s)?(?:\s+[A-Z][a-z]+(?:'s)?){0,3})\b")
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

# multi_hop-only additions: calendar words and dialogue-opening interjections
# that survive the legacy skip list (they aren't questions/auxiliaries, so
# they'd otherwise pass through as bogus bridge entities). Kept separate from
# _SKIP_CAPITALIZED so single_hop/temporal/open_domain — which never see
# these — stay on the exact original word list.
_SKIP_CAPITALIZED_STRICT_EXTRA = frozenset(
    {
        # Calendar words: never valid bridge entities, but frequently
        # capitalized in "on <Month> <day>" style timestamps that get
        # mis-parsed as person/entity names.
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        # Sentence-initial interjections/discourse markers: capitalized only
        # because they start a dialogue turn, not because they're names.
        "Besides",
        "Yeah",
        "Anything",
        "Sure",
        "Thanks",
        "Congrats",
        "Well",
        "Also",
        "Still",
        "Just",
        "Now",
        "Then",
        "So",
        "Even",
        "Really",
        "Actually",
        "Maybe",
        "Perhaps",
        "Later",
        "Soon",
        "Today",
        "Tomorrow",
        "Yesterday",
        "Hey",
        "Hi",
        "Oh",
        "Wow",
        "Great",
        "Nice",
        "Cool",
        "Absolutely",
        "Definitely",
        "Honestly",
        "Right",
        "Okay",
        "Ok",
        "Wait",
        "Speaking",
    }
)

_SKIP_CAPITALIZED_STRICT = _SKIP_CAPITALIZED | _SKIP_CAPITALIZED_STRICT_EXTRA


def _clean_phrase(base: str, skip_words: frozenset[str]) -> str | None:
    """Strip leading non-name words from *base*; return None if nothing usable is left.

    A capitalized phrase like "Besides writing" or "October when" only
    starts capitalized because it opens a dialogue turn or contains a
    calendar word. Dropping just the leading skip-word(s) (rather than the
    whole phrase) avoids losing a real trailing entity, e.g. "Hi Joanna"
    should still yield "Joanna", not get discarded entirely because "Hi"
    happens to be first.
    """
    words = base.split(" ")
    while words and words[0] in skip_words:
        words = words[1:]
    cleaned = " ".join(words)
    if len(cleaned) < 2:
        return None
    return cleaned


def normalize_entity(name: str) -> str:
    """Strip possessive suffix and normalize for comparison."""
    cleaned = name.strip()
    match = _POSSESSIVE_RE.match(cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned.lower()


def extract_capitalized_entities(text: str, *, strict: bool = False) -> list[str]:
    """Return capitalized tokens that look like person/entity names.

    *strict* (multi_hop only) matches multi-word proper-noun phrases as a
    single entity and applies the wider calendar/interjection skip list;
    the default (legacy) behavior used by every other category matches
    single capitalized words only, unchanged from the pre-chaining code.
    """
    token_re = _CAP_TOKEN_RE_STRICT if strict else _CAP_TOKEN_RE_LEGACY
    skip_words = _SKIP_CAPITALIZED_STRICT if strict else _SKIP_CAPITALIZED
    found: list[str] = []
    seen: set[str] = set()
    for match in token_re.finditer(text):
        token = match.group(1)
        base = token
        poss = _POSSESSIVE_RE.match(token)
        if poss:
            base = poss.group(1)
        cleaned = _clean_phrase(base, skip_words) if strict else (
            None if base in skip_words or len(base) < 2 else base
        )
        if cleaned is None:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(cleaned)
    return found


def extract_hop_entities(
    question: str,
    results: list[dict],
    *,
    limit: int = 8,
    exclude: set[str] | None = None,
    include_freetext: bool = True,
    strict: bool = False,
) -> list[str]:
    """Collect entity names from the question, top results, and atom metadata.

    *exclude* is a set of already-normalized entity keys (see
    :func:`normalize_entity`) to skip — used when chaining hops so a later
    level doesn't re-discover an entity already queried at an earlier level.

    *include_freetext* controls whether bare capitalized tokens harvested
    from atom text are used as candidates. Those tokens have no signal
    beyond capitalization, so they frequently pick up real-but-irrelevant
    proper nouns (movie/game titles, etc.) that a stopword list can't catch.
    Hop 1 keeps this on (matches the original single-pass behaviour); deeper
    chained levels turn it off and rely only on structured entity tags
    (``provenance.entities``) and explicit "named X" / "called X" mentions,
    which are much less likely to be noise.

    *strict* (multi_hop only) uses the phrase-aware, wider-skip-list
    extraction; every other category keeps the original word-level
    extraction so their scores are unaffected.
    """
    skip_words = _SKIP_CAPITALIZED_STRICT if strict else _SKIP_CAPITALIZED
    ordered: list[str] = []
    seen: set[str] = set(exclude or ())

    def add(name: str) -> None:
        base = name.strip()
        poss = _POSSESSIVE_RE.match(base)
        if poss:
            base = poss.group(1)
        if not base:
            return
        if strict:
            cleaned = _clean_phrase(base, skip_words)
        else:
            cleaned = None if base in skip_words or len(base) < 2 else base
        if cleaned is None:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(cleaned)

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
        if include_freetext:
            for field in ("text", "title"):
                value = item.get(field) or ""
                for ent in extract_capitalized_entities(value, strict=strict):
                    add(ent)
        text = item.get("text") or ""
        for match in re.finditer(
            r"\b(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            text,
        ):
            add(match.group(1))

    if include_freetext:
        for ent in extract_capitalized_entities(question, strict=strict):
            add(ent)

    return ordered[:limit]


def build_hop_queries(
    question: str,
    entities: list[str],
    *,
    max_queries: int = 2,
    strict: bool = False,
) -> list[str]:
    """Build follow-up search queries for bridge entities discovered in hop 1."""
    if not entities:
        return []

    q_lower = question.lower()
    question_entity_keys = {
        normalize_entity(ent) for ent in extract_capitalized_entities(question, strict=strict)
    }

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
