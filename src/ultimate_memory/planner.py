"""Generic query planning for memory retrieval.

This module is intentionally benchmark-agnostic. It infers retrieval requirements
from the question itself rather than accepting gold benchmark category labels.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .atoms import is_temporal_query


class QueryPlan(BaseModel):
    kind: Literal["single_hop", "multi_hop", "temporal"] = "single_hop"
    temporal_mode: Literal["current", "historical", "as_of", "unspecified"] = "unspecified"
    hop_depth: int = 1
    include_superseded: bool = False
    entities: list[str] = Field(default_factory=list)
    memory_types: list[str] = Field(default_factory=list)
    expansions: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    multi_evidence: bool = False
    requires_bridge: bool = False


_CAPITALIZED = re.compile(r"\b([A-Z][a-zA-Z0-9_.-]*(?:\s+[A-Z][a-zA-Z0-9_.-]*){0,3})\b")
_QUESTION_WORDS = {
    "What", "Where", "When", "Who", "Whom", "Which", "How", "Why", "Does",
    "Did", "Do", "Is", "Are", "Was", "Were", "Can", "Could", "Would", "Should",
}

_TEMPORAL_QUESTION_RE = re.compile(
    r"\bwhen\b|"
    r"\bwhat\s+(?:date|year|month|day)\b|"
    r"\bhow\s+long\b|"
    r"\bhow\s+many\s+(?:years?|months?|weeks?|days?|hours?)\b|"
    r"\b(?:years?|months?|weeks?|days?|hours?)\s+ago\b|"
    r"\b(?:since|until|during)\b|"
    r"\b(?:before|after|earlier|later|previously|formerly|prior)\b"
)

_COLLECTIVE_RE = re.compile(
    r"\bboth\b|"
    r"\bin\s+common\b|"
    r"\bshared\b|"
    r"\beach\b|"
    r"\brespectively\b|"
    r"\btogether\b|"
    r"\bcompare\b|"
    r"\bsimilarit(?:y|ies)\b|"
    r"\bdifferences?\b|"
    r"\bbetween\b.+\band\b",
    re.I,
)

_LIST_OR_SET_RE = re.compile(
    r"\bwhich\s+(?!(?:does|has|is|was|this)\b)(?:[a-z]+s|cities|places|countries|states|books|games|activities|items|things|ways|types|kinds)\b|"
    r"\bwhat\s+(?!(?:does|has|is|was|this)\b)(?:[a-z]+s|cities|places|countries|states|books|games|activities|items|things|ways|types|kinds)\b|"
    r"\bwhat\s+does\s+.+?\s+(?:offer|provide|include|do\s+to)\b|"
    r"\b(?:all|multiple|several)\s+(?:[a-z]+s|cities|places|books|activities|items|things|ways|types)\b|"
    r"\bwhere\s+has\s+.+?\s+(?:camped|traveled|travelled|visited|stayed|lived)\b|"
    r"\bwhat\s+.+?\s+has\s+.+?\s+(?:done|read|visited|attended|participated|painted|tried|used)\b",
    re.I,
)


def _is_temporal_question(question: str) -> bool:
    return bool(_TEMPORAL_QUESTION_RE.search(question.lower()))


def _is_collective_multi_hop(question: str, entities: list[str]) -> bool:
    lower = question.lower()
    if _COLLECTIVE_RE.search(lower) or _LIST_OR_SET_RE.search(lower):
        return True
    if re.search(
        r"\bhow\s+long\b.*\b(?:take|took|until|from|between|before|after)\b|"
        r"\b(?:duration|elapsed|time\s+between)\b",
        lower,
    ):
        return True
    if len(entities) >= 2 and re.search(r"\b(?:and|versus|vs\.?|compared?\s+to)\b", lower):
        return True
    return False

_RELATION_WORDS = {
    "sister", "brother", "mother", "father", "parent", "child", "children",
    "friend", "mentor", "manager", "boss", "employer", "company", "team",
    "project", "repository", "repo", "owner", "author", "maintainer",
}


def _entities(question: str) -> list[str]:
    found: list[str] = []
    for match in _CAPITALIZED.finditer(question):
        value = match.group(1).strip()
        if value.split()[0] in _QUESTION_WORDS:
            continue
        if value not in found:
            found.append(value)
    return found[:8]


def _requires_bridge(question: str) -> bool:
    """Whether answering requires traversing an unnamed relationship chain."""
    lower = question.lower()
    possessives = len(re.findall(r"\b[\w.-]+'s\b", question))
    relation_hits = sum(
        1 for word in _RELATION_WORDS
        if re.search(rf"\b{re.escape(word)}\b", lower)
    )
    chained_of = len(
        re.findall(r"\bof\s+(?:the\s+)?(?:\w+\s+){0,2}(?:of|for|at)\b", lower)
    )
    return possessives >= 2 or chained_of > 0 or (possessives >= 1 and relation_hits >= 1)


def _hop_depth(question: str, entities: list[str] | None = None) -> int:
    lower = question.lower()
    entities = entities or _entities(question)
    possessives = len(re.findall(r"\b[\w.-]+'s\b", question))
    relation_hits = sum(1 for word in _RELATION_WORDS if re.search(rf"\b{re.escape(word)}\b", lower))
    chained_of = len(re.findall(r"\bof\s+(?:the\s+)?(?:\w+\s+){0,2}(?:of|for|at)\b", lower))
    score = possessives + max(0, relation_hits - 1) + chained_of
    if _is_collective_multi_hop(question, entities):
        score = max(score, 2)
    if score >= 3:
        return 3
    if score >= 2:
        return 2
    return 1


def _memory_types(question: str) -> list[str]:
    lower = question.lower()
    types: list[str] = []
    if re.search(r"\bprefer|preference|favorite|favourite|always|never|style|likes?\b", lower):
        types.append("preference")
    if re.search(r"\bdecision|decide|decided|chose|chosen|why did we|why was\b", lower):
        types.append("decision")
    if re.search(r"\bhow do|how to|steps?|procedure|deploy|runbook|process\b", lower):
        types.append("procedure")
    if not types:
        types.append("fact")
    return types


def _expansions(question: str) -> list[str]:
    """Small generic synonym expansions; no dataset/entity-specific rules."""
    lower = question.lower()
    expansions: list[str] = []
    if re.search(r"\bjob|work|occupation|profession|employer\b", lower):
        expansions.append("work job employer role")
    if re.search(r"\blive|lives|location|based|where\b", lower):
        expansions.append("location lives based moved")
    if re.search(r"\bprefer|preference|favorite|favourite|likes?\b", lower):
        expansions.append("preference prefer favorite favourite likes")
    if re.search(r"\bdecision|decide|chose|chosen|why\b", lower):
        expansions.append("decision chose reason rationale")
    if re.search(r"\bhow do|how to|steps?|procedure|process\b", lower):
        expansions.append("procedure steps process")
    if re.search(r"\bbefore|previous|formerly|used to|prior\b", lower):
        expansions.append("previous formerly before historical")
    if re.search(r"\bvisit|visited|trip|travel|cities|places\b", lower):
        expansions.append("visited travel trip city place")
    if re.search(r"\boffer|offers|offering|provide|provides|services\b", lower):
        expansions.append("offer provides services classes workshops training")
    if _is_temporal_question(question):
        expansions.append("date year month day when duration time")
    if re.search(r"\bhow\s+long\b|\bduration\b|\belapsed\b", lower):
        expansions.append("started began finished completed opened duration elapsed")
    return list(dict.fromkeys(expansions))


def plan_query(question: str, *, as_of: str | None = None) -> QueryPlan:
    lower = question.lower()
    signals: list[str] = []

    if as_of:
        temporal_mode = "as_of"
        include_superseded = True
        signals.append("explicit_as_of")
    elif is_temporal_query(question) or re.search(
        r"\b(before|previously|formerly|used to|prior|back when|at the time)\b", lower
    ):
        temporal_mode = "historical"
        include_superseded = True
        signals.append("historical_language")
    elif re.search(r"\b(now|current|currently|today|latest)\b", lower):
        temporal_mode = "current"
        include_superseded = False
        signals.append("current_language")
    else:
        temporal_mode = "unspecified"
        include_superseded = False

    entities = _entities(question)
    multi_evidence = _is_collective_multi_hop(question, entities)
    requires_bridge = _requires_bridge(question)
    depth = _hop_depth(question, entities)
    temporal_question = _is_temporal_question(question)
    if depth > 1:
        kind = "multi_hop"
        signals.append(f"relation_chain_depth_{depth}")
    elif temporal_mode in {"historical", "as_of"} or temporal_question:
        kind = "temporal"
    else:
        kind = "single_hop"

    return QueryPlan(
        kind=kind,
        temporal_mode=temporal_mode,
        hop_depth=depth,
        include_superseded=include_superseded,
        entities=entities,
        memory_types=_memory_types(question),
        expansions=_expansions(question),
        signals=signals,
        multi_evidence=multi_evidence,
        requires_bridge=requires_bridge,
    )
