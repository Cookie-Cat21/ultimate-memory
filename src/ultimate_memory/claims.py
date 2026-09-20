"""Structured claim extraction and conflict resolution.

Claims are stored in AtomicMemory.metadata so the existing SQLite schema remains
backwards compatible. The claim layer is deliberately generic and never depends
on benchmark entities or categories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AtomicMemory

_SINGLE_VALUE_PREDICATES = {
    "location.current",
    "employment.role",
    "employment.employer",
    "project.database",
    "project.region",
    "project.runtime",
    "project.package_manager",
    "preference.primary",
}

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?P<s>.+?)\s+(?:currently\s+)?(?:lives?|is living|is based|is located)\s+in\s+(?P<o>.+?)[.!]?$", re.I), "location.current"),
    (re.compile(r"^(?P<s>.+?)\s+(?:moved|relocated)\s+to\s+(?P<o>.+?)(?:\s+instead\b.*)?[.!]?$", re.I), "location.current"),
    (re.compile(r"^(?P<s>.+?)\s+(?:works?|is working|is employed)\s+as\s+(?P<o>.+?)(?:\s+at\b.*)?[.!]?$", re.I), "employment.role"),
    (re.compile(r"^(?P<s>.+?)\s+(?:works?|is working|is employed)\s+(?:at|for)\s+(?P<o>.+?)[.!]?$", re.I), "employment.employer"),
    (re.compile(r"^(?P<s>.+?)\s+(?:prefers?|would rather use)\s+(?P<o>.+?)[.!]?$", re.I), "preference.primary"),
    (re.compile(r"^(?P<s>.+?)\s+(?:uses?|is using|runs?)\s+(?P<o>PostgreSQL|MySQL|SQLite|MongoDB|Redis)\b.*$", re.I), "project.database"),
    (re.compile(r"^(?P<s>.+?)\s+(?:uses?|is using)\s+(?P<o>npm|pnpm|yarn|bun)\b.*$", re.I), "project.package_manager"),
    (re.compile(r"^(?P<s>.+?)\s+(?:deploys?|is deployed|runs?)\s+(?:in|on)\s+(?P<o>[a-z]{2}(?:-[a-z0-9]+){1,3}|[A-Za-z]+\s+region)\b.*$", re.I), "project.region"),
)


def _clean_subject(value: str) -> str:
    value = re.sub(r"^(?:fact|decision|preference|procedure)\s*:\s*", "", value.strip(), flags=re.I)
    if value.lower() == "i":
        return "user"
    if value.lower() == "we":
        return "team"
    return value.strip(" .,:;-")


def _clean_object(value: str) -> str:
    value = re.split(
        r"\b(?:instead of|rather than|as opposed to|because|after moving from|formerly)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    return value.strip(" .,:;-\"'")


@dataclass(frozen=True)
class StructuredClaim:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.82

    def as_dict(self) -> dict:
        return {
            "schema": "ultimate-memory.claim.v1",
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
        }


def extract_claim(text: str) -> StructuredClaim | None:
    compact = re.sub(r"\s+", " ", text.strip())
    for pattern, predicate in _PATTERNS:
        match = pattern.match(compact)
        if not match:
            continue
        subject = _clean_subject(match.group("s"))
        obj = _clean_object(match.group("o"))
        if len(subject) < 1 or len(obj) < 1:
            continue
        return StructuredClaim(subject=subject, predicate=predicate, object=obj)
    return None


def claim_from_atom(atom: AtomicMemory) -> StructuredClaim | None:
    raw = atom.metadata.get("claim")
    if isinstance(raw, dict):
        subject = str(raw.get("subject") or "").strip()
        predicate = str(raw.get("predicate") or "").strip()
        obj = str(raw.get("object") or "").strip()
        if subject and predicate and obj:
            return StructuredClaim(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=float(raw.get("confidence") or 0.82),
            )
    return extract_claim(atom.text)


def ensure_claim_metadata(atom: AtomicMemory) -> StructuredClaim | None:
    claim = claim_from_atom(atom)
    if claim is not None:
        atom.metadata["claim"] = claim.as_dict()
    return claim


def structured_conflict_score(new_atom: AtomicMemory, old_atom: AtomicMemory) -> float:
    new_claim = claim_from_atom(new_atom)
    old_claim = claim_from_atom(old_atom)
    if not new_claim or not old_claim:
        return 0.0
    if new_claim.subject.casefold() != old_claim.subject.casefold():
        return 0.0
    if new_claim.predicate != old_claim.predicate:
        return 0.0
    if new_claim.object.casefold() == old_claim.object.casefold():
        return 0.0
    if new_claim.predicate in _SINGLE_VALUE_PREDICATES:
        return min(0.98, 0.82 + 0.08 * (new_claim.confidence + old_claim.confidence) / 2)
    return 0.0
