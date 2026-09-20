"""Entity-chain evidence reranking for multi-hop memory questions."""
from __future__ import annotations

import re
from collections import deque

from .hops import extract_capitalized_entities, normalize_entity

_STOP = frozenset({
    "what", "where", "when", "who", "whom", "which", "how", "why",
    "does", "did", "do", "is", "are", "was", "were", "has", "have", "had",
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "from", "at",
    "its", "his", "her", "their", "this", "that", "these", "those",
})


def _question_terms(question: str, entity_keys: set[str]) -> list[str]:
    entity_tokens = {part for key in entity_keys for part in key.split()}
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower()):
        if token in _STOP or token in entity_tokens or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:10]


def _context_entities(item: dict) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    provenance = item.get("provenance") or {}
    for raw in provenance.get("entities") or []:
        if not isinstance(raw, str):
            continue
        key = normalize_entity(raw)
        if key and key not in seen:
            seen.add(key)
            found.append(raw)
    for raw in extract_capitalized_entities(str(item.get("text") or ""), strict=True):
        key = normalize_entity(raw)
        if key and key not in seen:
            seen.add(key)
            found.append(raw)
    return found[:16]


def rank_evidence_chain(question: str, contexts: list[dict], *, max_depth: int = 4) -> list[dict]:
    """Rerank evidence by graph reachability from question entities.

    Each context contributes co-occurrence edges between the entities it contains.
    Contexts that lie on a reachable chain and also mention the final relation
    terms from the question rise above merely lexical matches.
    """
    if not contexts:
        return []

    question_entities = extract_capitalized_entities(question, strict=True)
    q_keys = {normalize_entity(entity) for entity in question_entities if normalize_entity(entity)}
    if not q_keys:
        return list(contexts)

    entities_by_index: list[list[str]] = []
    adjacency: dict[str, set[str]] = {}
    for item in contexts:
        keys = [normalize_entity(e) for e in _context_entities(item)]
        keys = [k for k in dict.fromkeys(keys) if k]
        entities_by_index.append(keys)
        for key in keys:
            adjacency.setdefault(key, set())
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    distance: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((key, 0) for key in q_keys)
    while queue:
        key, depth = queue.popleft()
        if key in distance and distance[key] <= depth:
            continue
        distance[key] = depth
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(key, ()):
            queue.append((neighbor, depth + 1))

    terms = _question_terms(question, q_keys)
    ranked: list[dict] = []
    for item, keys in zip(contexts, entities_by_index, strict=True):
        reachable = [distance[key] for key in keys if key in distance]
        min_depth = min(reachable) if reachable else None
        max_reachable_depth = max(reachable) if reachable else 0
        text = str(item.get("text") or "").lower()
        target_hits = sum(1 for term in terms if term in text)
        bridge_count = len([key for key in keys if key in distance])

        score = float(item.get("score") or 0.0)
        if min_depth is not None:
            score += 0.34
            score += max(0.0, 0.16 - 0.04 * min_depth)
            score += min(0.18, 0.05 * max_reachable_depth)
        score += min(0.36, 0.09 * target_hits)
        if bridge_count >= 2:
            score += 0.08

        enriched = dict(item)
        provenance = dict(enriched.get("provenance") or {})
        provenance["chain_reachable"] = min_depth is not None
        provenance["chain_min_depth"] = min_depth
        provenance["chain_max_depth"] = max_reachable_depth
        provenance["chain_target_hits"] = target_hits
        provenance["chain_score"] = round(score, 6)
        enriched["provenance"] = provenance
        enriched["score"] = score
        ranked.append(enriched)

    ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked
