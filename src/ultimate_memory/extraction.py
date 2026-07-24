"""Heuristic session→memory extraction for conversational transcripts.

Pure-Python patterns tuned for LoCoMo-style speaker-tagged dialogue. No LLM APIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .dates import parse_loose_date, resolve_relative_dates
from .models import ReflectionPayload

# Legacy keyword signals (fallback merge).
_DECISION_SIGNALS = (
    "decided to",
    "decision:",
    "we chose",
    "going with",
    "will use",
    "switching to",
    "the approach is",
)
_PREFERENCE_SIGNALS = (
    "prefer ",
    "preference:",
    "always ",
    "never ",
    "i like",
    "i don't like",
    "i want",
    "please avoid",
)
_PROCEDURE_SIGNALS = (
    "steps:",
    "to do:",
    "procedure:",
    "how to ",
    "the process is",
    "to fix this",
)
_FACT_SIGNALS = (
    "the fix was",
    "the issue was",
    "root cause",
    "turns out",
    "note that",
    "important:",
    "the problem is",
    "bug:",
)

_MIN_LEN = 20
_MAX_LEN = 220

# Markdown-style labels that look like "Speaker:" but are not dialogue tags.
_LABEL_PREFIXES = frozenset(
    {
        "steps",
        "step",
        "note",
        "notes",
        "important",
        "decision",
        "procedure",
        "todo",
        "warning",
        "context",
        "summary",
        "tags",
        "created_at",
        "session_date",
        "date",
        "event_time",
    }
)

_SPEAKER_LINE = re.compile(
    r"^(?P<speaker>[A-Z][a-zA-Z][\w.-]{0,30}):\s*(?P<utterance>.+)$"
)
_SPEAKER_SAID = re.compile(
    r"(?P<speaker>[A-Z][a-zA-Z][\w.-]{0,30})\s+said\s+(?:she|he|they|that)\s+(?P<utterance>.+)",
    re.IGNORECASE,
)
_DIALOGUE_TURN_LINE = re.compile(r"^\[(D\d+:\d+)\]\s*([^:]+):\s*(.+)$")
_MAX_TURN_ATOMS = 40
_MIN_TURN_ATOM_LEN = 20

# Fact patterns: (compiled regex, optional speaker prefix template)
_FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:lives?|living)\s+in\s+.+", re.I), "{speaker} lives in {match}"),
    (re.compile(r"\b(?:moved|moving)\s+to\s+.+", re.I), "{speaker} moved to {match}"),
    (re.compile(r"\b(?:works?|working)\s+as\s+(?:a\s+)?.+", re.I), "{speaker} works as {match}"),
    (re.compile(r"\bemployed\s+as\s+(?:a\s+)?.+", re.I), "{speaker} is employed as {match}"),
    (re.compile(r"\b(?:has|have)\s+a\s+(?:dog|cat|pet|puppy|kitten)\b.+", re.I), "{text}"),
    (re.compile(r"\ballergic\s+to\s+.+", re.I), "{speaker} is allergic to {match}"),
    (re.compile(r"\b(?:started|beginning)\s+(?:working|dating|studying|training)\b.+", re.I), "{text}"),
    (re.compile(r"\bgraduated\s+from\s+.+", re.I), "{speaker} graduated from {match}"),
    (re.compile(r"\b(?:got\s+)?married\s+(?:to\s+.+|last\s+.+)", re.I), "{text}"),
    (re.compile(r"\bbirthday\s+is\s+.+", re.I), "{speaker}'s birthday is {match}"),
    (re.compile(r"\bborn\s+(?:on|in)\s+.+", re.I), "{speaker} was born {match}"),
    (re.compile(r"\bmet\s+(?:at|in|on|during)\s+.+", re.I), "{speaker} met someone {match}"),
    (re.compile(r"\b(?:i'?m|i\s+am)\s+a[n]?\s+.+", re.I), "{text}"),
    (re.compile(r"\bmy\s+(?:sister|brother|mother|father|mom|dad|wife|husband|partner|son|daughter|friend)\b.+", re.I), "{text}"),
    (re.compile(r"\b(?:from|based\s+in)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b", re.I), "{text}"),
    (re.compile(r"\b(?:diagnosed\s+with|suffers?\s+from|dealing\s+with)\s+.+", re.I), "{text}"),
    (re.compile(r"\b(?:celebrated|attended|visited|traveled\s+to)\s+.+", re.I), "{text}"),
    (
        re.compile(
            r"\b(?:went|go|going)\s+to\s+.+\b(?:on|in)\s+"
            r"(?:\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4})\b.*",
            re.I,
        ),
        "{speaker} {match}",
    ),
    (
        re.compile(
            r"\b(?:on|in)\s+(?:\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4})\b.+",
            re.I,
        ),
        "{text}",
    ),
    (re.compile(r"\b(?:i'?m|i\s+am)\s+(?:a\s+)?transgender\b.+", re.I), "{speaker} is {match}"),
    (re.compile(r"\bresearch(?:ing|ed)?\s+.+", re.I), "{speaker} researched {match}"),
    (re.compile(r"\bpainted\s+(.+)", re.I), "{speaker} painted {match}"),
]

_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:i\s+)?(?:really\s+)?(?:love|like|enjoy|prefer)\s+.+", re.I),
    re.compile(r"\b(?:don'?t|do\s+not)\s+like\s+.+", re.I),
    re.compile(r"\b(?:favorite|favourite)\s+.+", re.I),
    re.compile(r"\b(?:always|never)\s+(?:use|eat|drink|watch|read|go|buy|wear)\b.+", re.I),
    re.compile(r"\b(?:hate|dislike|avoid)\s+.+", re.I),
]

_DECISION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:decided|cho(?:se|sen)|picked|opted)\s+(?:to\s+)?.+", re.I),
    re.compile(r"\b(?:going\s+to|will|plan\s+to|plans\s+to)\s+(?:buy|move|switch|use|adopt|start)\b.+", re.I),
    re.compile(r"\b(?:we\s+)?(?:chose|chosen|settled\s+on)\s+.+", re.I),
]

_PROCEDURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhow\s+to\s+.+", re.I),
    re.compile(r"\b(?:first|then|next|finally)\b.+(?:\bthen\b|\bfinally\b)", re.I),
    re.compile(r"\b(?:steps?|process)\s*(?::|is|are)\s*.+", re.I),
    re.compile(r"\bto\s+(?:fix|install|setup|set\s+up|configure|deploy)\b.+", re.I),
]

_ENTITY_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_RELATIONSHIP_ENTITY = re.compile(
    r"\b(?:married\s+to|met|dating|friend|sister|brother|mother|father|"
    r"partner|colleague|boss|neighbor|neighbour)\s+([A-Z][a-z]+)\b",
    re.I,
)
_SKIP_ENTITY = frozenset(
    {
        "I",
        "We",
        "The",
        "This",
        "That",
        "What",
        "When",
        "Where",
        "How",
        "Why",
        "Yes",
        "No",
        "Hi",
        "Hello",
        "Thanks",
        "Thank",
        "Great",
        "Good",
        "Well",
        "Oh",
        "Okay",
        "Ok",
        "Sure",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)


@dataclass
class ExtractionResult:
    """Reflection payload plus extracted entity names for graph linking."""

    payload: ReflectionPayload | None
    entities: list[str] = field(default_factory=list)


@dataclass
class ParsedDialogueTurn:
    dia_id: str
    speaker: str
    utterance: str

    @property
    def line_text(self) -> str:
        return f"[{self.dia_id}] {self.speaker}: {self.utterance}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:_MAX_LEN]


def _complete_enough(text: str) -> bool:
    norm = _normalize(text)
    return _MIN_LEN <= len(norm) <= _MAX_LEN


def _parse_line(raw_line: str) -> tuple[str | None, str]:
    """Return (speaker, utterance) for a transcript line."""
    line = raw_line.strip()
    if not line:
        return None, ""

    m = _SPEAKER_LINE.match(line)
    if m:
        speaker = m.group("speaker")
        if speaker.lower() not in _LABEL_PREFIXES:
            return speaker, m.group("utterance").strip()
        return None, line

    m = _SPEAKER_SAID.search(line)
    if m:
        return m.group("speaker"), m.group("utterance").strip()

    return None, line


def _with_speaker(speaker: str | None, text: str) -> str:
    if not speaker:
        return text
    lower = text[:1].lower() + text[1:] if text else text
    if lower.startswith(("i ", "i'", "my ", "we ")):
        return f"{speaker}: {text}"
    return f"{speaker} {lower}"


def _fill_template(template: str, *, speaker: str | None, text: str, match: str) -> str:
    out = template.format(speaker=speaker or "They", text=text, match=match.strip(" ."))
    return _with_speaker(speaker, out) if "{speaker}" not in template and speaker else out


def _extract_entities_from_line(speaker: str | None, text: str) -> list[str]:
    found: list[str] = []
    if speaker:
        found.append(speaker)
    for m in _RELATIONSHIP_ENTITY.finditer(text):
        found.append(m.group(1))
    for m in _ENTITY_NAME.finditer(text):
        name = m.group(1)
        parts = name.split()
        if all(p not in _SKIP_ENTITY for p in parts):
            found.append(name)
    return found


def _match_fact(speaker: str | None, text: str) -> str | None:
    for pattern, template in _FACT_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        matched = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
        if template == "{text}":
            candidate = _with_speaker(speaker, text) if speaker else text
        else:
            candidate = _fill_template(template, speaker=speaker, text=text, match=matched)
        if _complete_enough(candidate):
            return candidate
    return None


def _match_patterns(speaker: str | None, text: str, patterns: list[re.Pattern[str]]) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            candidate = _with_speaker(speaker, text) if speaker else text
            if _complete_enough(candidate):
                return candidate
    return None


def _keyword_bucket(lower: str) -> str | None:
    if any(sig in lower for sig in _DECISION_SIGNALS):
        return "decisions"
    if any(sig in lower for sig in _PREFERENCE_SIGNALS):
        return "preferences"
    if any(sig in lower for sig in _PROCEDURE_SIGNALS):
        return "procedures"
    if any(sig in lower for sig in _FACT_SIGNALS):
        return "facts"
    return None


def _add_unique(bucket: list[str], seen: set[str], line: str) -> None:
    norm = _normalize(line)
    key = norm.lower()
    if norm and key not in seen and _complete_enough(norm):
        seen.add(key)
        bucket.append(norm)


def parse_dialogue_turns(text: str) -> list[ParsedDialogueTurn]:
    """Parse LoCoMo-style dialogue lines like ``[D1:3] Caroline: text``."""
    turns: list[ParsedDialogueTurn] = []
    for raw_line in text.splitlines():
        match = _DIALOGUE_TURN_LINE.match(raw_line.strip())
        if not match:
            continue
        turns.append(
            ParsedDialogueTurn(
                dia_id=match.group(1),
                speaker=match.group(2).strip(),
                utterance=match.group(3).strip(),
            )
        )
    return turns


def session_anchor_from_text(text: str):
    """Parse session date from transcript frontmatter if present."""
    return _session_anchor(text)


def _session_anchor(text: str):
    """Parse session date from transcript frontmatter if present."""
    for line in text.splitlines()[:40]:
        lower = line.strip().lower()
        for key in ("created_at:", "session_date:", "date:", "event_time:"):
            if lower.startswith(key):
                value = line.split(":", 1)[1].strip().strip('"').strip("'")
                parsed = parse_loose_date(value)
                if parsed:
                    return parsed
    return None


def extract_from_transcript(
    text: str,
    session_id: str,
    project_path: str | None,
) -> ExtractionResult:
    """Extract facts, preferences, decisions, procedures, and entities from a transcript."""
    facts: list[str] = []
    decisions: list[str] = []
    preferences: list[str] = []
    procedures: list[str] = []
    entities: list[str] = []
    seen: set[str] = set()
    seen_entities: set[str] = set()
    anchor = _session_anchor(text)

    def _track_entities(speaker: str | None, utterance: str) -> None:
        for name in _extract_entities_from_line(speaker, utterance):
            key = name.lower()
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append(name)

    for raw_line in text.splitlines():
        # Strip dialogue-id prefixes like [D1:3]
        cleaned = re.sub(r"^\[D\d+:\d+\]\s*", "", raw_line.strip())
        speaker, utterance = _parse_line(cleaned)
        if not utterance:
            continue

        utterance = resolve_relative_dates(utterance, anchor)
        _track_entities(speaker, utterance)
        lower = utterance.lower()
        categorized = False

        if fact := _match_fact(speaker, utterance):
            _add_unique(facts, seen, fact)
            categorized = True
        elif dec := _match_patterns(speaker, utterance, _DECISION_PATTERNS):
            _add_unique(decisions, seen, dec)
            categorized = True
        elif pref := _match_patterns(speaker, utterance, _PREFERENCE_PATTERNS):
            _add_unique(preferences, seen, pref)
            categorized = True
        elif proc := _match_patterns(speaker, utterance, _PROCEDURE_PATTERNS):
            _add_unique(procedures, seen, proc)
            categorized = True

        if not categorized:
            bucket = _keyword_bucket(lower)
            if bucket == "decisions":
                _add_unique(decisions, seen, _with_speaker(speaker, utterance))
            elif bucket == "preferences":
                _add_unique(preferences, seen, _with_speaker(speaker, utterance))
            elif bucket == "procedures":
                _add_unique(procedures, seen, _with_speaker(speaker, utterance))
            elif bucket == "facts":
                _add_unique(facts, seen, _with_speaker(speaker, utterance))

    facts = facts[:6]
    decisions = decisions[:6]
    preferences = preferences[:4]
    procedures = procedures[:4]
    entities = entities[:12]

    total_signals = len(facts) + len(decisions) + len(preferences) + len(procedures)
    if total_signals < 1:
        return ExtractionResult(payload=None, entities=entities)

    project_name = Path(project_path).name if project_path else "session"
    summary = (
        f"Auto-extracted from {project_name} session {session_id[:20]}. "
        f"{total_signals} signals: {len(decisions)} decisions, {len(facts)} facts, "
        f"{len(preferences)} preferences, {len(procedures)} procedures."
    )
    if entities:
        summary += f" Entities: {', '.join(entities[:5])}."

    payload = ReflectionPayload(
        summary=summary,
        facts=facts,
        decisions=decisions,
        preferences=preferences,
        procedures=procedures,
        source_refs=[f"session:{session_id}"],
    )
    return ExtractionResult(payload=payload, entities=entities)
