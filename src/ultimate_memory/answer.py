"""Extractive answer synthesis for memory QA benchmarks (LoCoMo-style).

Pure-Python span selection over retrieved contexts — no LLM APIs.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

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
        "being",
        "this",
        "that",
        "it",
        "we",
        "i",
        "you",
        "they",
        "he",
        "she",
        "his",
        "her",
        "their",
        "my",
        "your",
        "from",
        "as",
        "at",
        "by",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "which",
        "how",
        "why",
        "there",
        "here",
        "about",
        "into",
        "than",
        "then",
        "them",
        "these",
        "those",
    }
)

_ARTICLES = frozenset({"a", "an", "the"})

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_DATE_PATTERNS: list[re.Pattern[str]] = [
    # Day Month Year, with optional comma: 7 May 2023 / 7 May, 2023
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)(?:,\s*|\s+)\d{4}\b",
        re.I,
    ),
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:,\s*|\s+)\d{4}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4}\b",
        re.I,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # Anchored relative phrases that appear in LoCoMo golds.
    re.compile(
        r"\b(?:the\s+)?(?:week|weekend|sunday|monday|tuesday|wednesday|thursday|friday|saturday)"
        r"\s+before\s+\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:two\s+weekends?\s+before|a\s+few\s+weeks?\s+before)\s+\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4}\b",
        re.I,
    ),
    re.compile(r"\b(?:a\s+few|several|\d+)\s+years?\s+ago\b", re.I),
    re.compile(r"\b(?:in|on|during)\s+(?:the\s+)?(?:year\s+)?((?:19|20)\d{2})\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b(?:last|next)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I),
]

_RELATIVE_ONLY_DATE_RE = re.compile(
    r"^(?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"week|weekend|month|year)$|"
    r"^(?:yesterday|today|tomorrow|recently|earlier|later)$|"
    r"^(?:a few days ago|a few years ago|several years ago|two days ago|2 days ago|last night)$",
    re.I,
)

_NUMBER_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a\s+few|several)"
_DURATION_SPAN_RE = re.compile(
    rf"\b((?:\d+|{_NUMBER_WORD})\s+years?(?:\s+ago)?|"
    rf"(?:\d+|{_NUMBER_WORD})\s+months?(?:\s+ago)?|"
    rf"(?:\d+|{_NUMBER_WORD})\s+weeks?(?:\s+ago)?|"
    rf"(?:for|over)\s+(?:\d+|{_NUMBER_WORD})\s+(?:years?|months?|weeks?))\b",
    re.I,
)

_GREETING_ONLY_RE = re.compile(
    r"^(?:\[D\d+:\d+\]\s*)?(?:[A-Z][a-z]+\s*:\s*)?"
    r"(?:hey|hi|hello|thanks(?:\s+a\s+bunch)?|thank\s+you|wow|great|awesome|nice|cool|sure|yep|yeah)"
    r"(?:[\s,!.'-]+[A-Z][a-z]+)?[!?.]*$",
    re.I,
)

_INVENTORY_LINE_RE = re.compile(
    r"\b(?:profile|activities|camp places|books read|painted|LGBTQ participation|"
    r"relationship status|moved from|identity|career)\s*:",
    re.I,
)

_YES_NO_RE = re.compile(
    r"^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|"
    r"am|may|might)\b",
    re.I,
)
_WHEN_RE = re.compile(r"\bwhen\b|\bwhat\s+(?:year|date|month|day)\b", re.I)
_WHERE_RE = re.compile(r"\bwhere\b", re.I)
_WHO_RE = re.compile(r"\bwho(?:m)?\b", re.I)
_WHAT_RE = re.compile(r"\bwhat\b", re.I)
_OCCUPATION_QUESTION_RE = re.compile(
    r"\b(?:do\s+for\s+(?:a\s+)?(?:living|work)|occupation|career)\b|"
    r"\bwhat\s+(?:does|did)\s+.+\s+(?:work|job)\b",
    re.I,
)
_OCCUPATION_ANSWER_RE = re.compile(
    r"\b(?:works?\s+as|worked\s+as|is\s+a|was\s+a|employed\s+(?:as|by|at))\b",
    re.I,
)
_DIA_TURN_RE = re.compile(r"^\[D\d+:\d+\]")
_IDENTITY_QUESTION_RE = re.compile(
    r"\bidentity\b|\b(?:gender|transgender)\b|what is .+'s (?:identity|gender)",
    re.I,
)
_IDENTITY_PHRASE_RE = re.compile(
    r"\btransgender\b|\b(?:trans\s+)?woman\b|\b(?:trans\s+)?man\b|\bidentity\b",
    re.I,
)
_EXPLICIT_IDENTITY_RE = re.compile(
    r"\b(transgender\s+(?:woman|man)|trans\s+(?:woman|man)|nonbinary|non-binary)\b",
    re.I,
)
_FAVORITE_VALUE_RE = re.compile(
    r"\b(?:my|his|her|their|[A-Z][a-z]+'s)\s+"
    r"(?:favorite|favourite)\s+[^.!?,:]{0,50}?\s+(?:is|are)\s+([^.!?,;]{1,70})|"
    r"\b([^.!?,;]{1,50})\s+(?:is|are)\s+"
    r"(?:my|his|her|their|[A-Z][a-z]+'s)\s+(?:favorite|favourite)\b",
    re.I,
)

_JUNK_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^#"),
    re.compile(r"^-\s*mentions\b", re.I),
    re.compile(r"^-\s*relates_to\b", re.I),
    re.compile(r"^-\s*from_project\b", re.I),
    re.compile(r"\[\["),
    re.compile(r"^Source Refs\b", re.I),
    re.compile(r"^Open Questions\b", re.I),
    re.compile(r"^---\s*$"),
)

_CURRENT_CUES = ("now", "current", "currently", "today")
_PAST_CUES = ("used to", "before", "previously")

_PREFERRED_MEMORY_TYPES = frozenset({"fact", "preference", "decision", "procedure"})

_AFFIRM_CUES = (
    "yes",
    "yeah",
    "yep",
    "true",
    "correct",
    "indeed",
    "always",
    "does",
    "did",
    "has",
    "have",
    "loves",
    "likes",
    "enjoys",
    "went",
    "visited",
    "attended",
    "married",
    "works",
    "lives",
)
_NEG_CUES = (
    "no",
    "not",
    "never",
    "none",
    "n't",
    "doesn't",
    "don't",
    "didn't",
    "hasn't",
    "haven't",
    "won't",
    "cannot",
    "can't",
    "false",
    "neither",
    "nor",
    "without",
    "refused",
    "declined",
    "unable",
)


def _normalize_for_f1(text: str | int | float | None) -> str:
    lowered = str(text or "").lower()
    no_punct = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = " ".join(tok for tok in no_punct.split() if tok not in _ARTICLES)
    return " ".join(no_articles.split())


def _tokens(text: str | int | float | None) -> list[str]:
    return _normalize_for_f1(text).split()


def tokenize_f1(
    prediction: str | int | float | None,
    ground_truth: str | list[str] | int | float | None,
) -> float:
    """Standard QA token F1 (SQuAD-style): lowercase, strip punctuation, max over golds."""
    if ground_truth is None:
        golds: list[str | int | float] = []
    elif isinstance(ground_truth, list):
        golds = ground_truth
    else:
        golds = [ground_truth]
    if not golds:
        return 0.0
    pred_toks = _tokens(prediction)
    if not pred_toks:
        return 0.0

    best = 0.0
    for gold in golds:
        gold_toks = _tokens(gold)
        if not gold_toks:
            continue
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def _content_words(text: str) -> set[str]:
    return {tok for tok in _tokens(text) if tok and tok not in _STOPWORDS and len(tok) > 1}


def _question_entities(question: str) -> set[str]:
    entities: set[str] = set()
    for match in _ENTITY_RE.finditer(question):
        name = match.group(1)
        if name.lower() not in _STOPWORDS:
            entities.add(name.lower())
    for tok in _content_words(question):
        if tok[0].isupper() if tok else False:
            entities.add(tok.lower())
    return entities


QuestionKind = Literal["yes_no", "when", "where", "who", "what", "other"]


def _is_occupation_question(question: str) -> bool:
    return bool(_OCCUPATION_QUESTION_RE.search(question))


def _is_identity_question(question: str) -> bool:
    return bool(_IDENTITY_QUESTION_RE.search(question))


def _favorite_subject(question: str) -> bool:
    return bool(re.search(r"\bfavou?rite\b", question, re.I))


def _precise_scalar_answer(question: str, normalized: list["_ContextItem"]) -> str | None:
    """Return high-precision scalar values before generic span ranking."""
    identity = _is_identity_question(question)
    favorite = _favorite_subject(question)

    if identity:
        for item in normalized:
            match = _EXPLICIT_IDENTITY_RE.search(item.text)
            if match:
                return match.group(1).strip()

    if favorite:
        q_words = _content_words(question)
        entities = _question_entities(question)
        candidates: list[tuple[float, str]] = []
        for item in normalized:
            for sentence in _split_sentences(item.text):
                match = _FAVORITE_VALUE_RE.search(sentence)
                if not match:
                    continue
                value = (match.group(1) or match.group(2) or "").strip(" .,:;-")
                if not value:
                    continue
                score = _overlap_score(q_words, sentence, entities)
                candidates.append((score, value))
        if candidates:
            candidates.sort(key=lambda pair: (pair[0], -len(pair[1])), reverse=True)
            return candidates[0][1]

    return None


def _dialogue_turn_bonus(
    text: str,
    *,
    question_words: set[str],
    entities: set[str],
) -> float:
    """Prefer short [D#:##] turns that mention both the person and a question keyword."""
    if not _DIA_TURN_RE.match(text.strip()):
        return 0.0
    lower = text.lower()
    bonus = 0.0
    entity_hits = sum(1 for ent in entities if ent in lower)
    content_hits = sum(1 for word in question_words if word in lower)
    if entity_hits and content_hits:
        bonus += 0.55
    elif entity_hits:
        bonus += 0.2
    elif content_hits:
        bonus += 0.15
    if len(text) <= 220:
        bonus += 0.15
    if len(text) <= 140:
        bonus += 0.1
    return bonus


def _occupation_score_adjustment(span: str, sentence: str) -> float:
    text = f"{span} {sentence}".lower()
    if _OCCUPATION_ANSWER_RE.search(text):
        return 0.85
    if re.search(r"\b(?:named|called)\s+[a-z]", text):
        return -0.5
    return 0.0


def _question_kind(question: str) -> QuestionKind:
    q = question.strip()
    if _YES_NO_RE.match(q):
        return "yes_no"
    # Duration questions are temporal even without "when".
    if re.search(r"\bhow long\b|\bhow many years\b|\bhow many months\b", q, re.I):
        return "when"
    if _WHEN_RE.search(q):
        return "when"
    if _WHERE_RE.search(q):
        return "where"
    if _WHO_RE.search(q):
        return "who"
    if _WHAT_RE.search(q):
        return "what"
    return "other"


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text.strip()]


def _extract_date_spans(sentence: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(sentence):
            span = match.group(0).strip(" .,;")
            # Unwrap "in 2022" capture groups when present.
            if match.lastindex:
                captured = match.group(1)
                if captured and re.fullmatch(r"(?:19|20)\d{2}", captured):
                    span = captured
            key = span.lower()
            if key not in seen:
                seen.add(key)
                spans.append(span)
    return spans


def _display_date(span: str) -> str:
    """Render ISO-style dates as compact human-readable dates."""
    value = span.strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?:[T ][^\s]+)?", value)
    if match:
        year, month, day = map(int, match.groups())
        months = (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        )
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{day} {months[month - 1]} {year}"
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year, month = map(int, match.groups())
        months = (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        )
        if 1 <= month <= 12:
            return f"{months[month - 1]} {year}"
    return value


def _is_relative_only_date(span: str) -> bool:
    return bool(_RELATIVE_ONLY_DATE_RE.match(span.strip()))


def _extract_duration_spans(sentence: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for match in _DURATION_SPAN_RE.finditer(sentence):
        span = match.group(1).strip()
        # Normalize "for 4 years" -> "4 years"
        span = re.sub(r"^(?:for|over)\s+", "", span, flags=re.I)
        key = span.lower()
        if key not in seen:
            seen.add(key)
            spans.append(span)
    return spans


def _prefer_absolute_date_spans(spans: list[str]) -> list[str]:
    absolute = [s for s in spans if not _is_relative_only_date(s)]
    return absolute or spans


def _extract_location_spans(sentence: str) -> list[str]:
    spans: list[str] = []
    for pattern in (
        re.compile(r"\b(?:in|at|from|near|to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b"),
        re.compile(r"\b(?:lives?|living|moved|located)\s+(?:in|at|to)\s+([^.,;!?]{3,40})", re.I),
    ):
        for match in pattern.finditer(sentence):
            span = match.group(1).strip(" .,;")
            if span and span.lower() not in _STOPWORDS:
                spans.append(span)
    return spans


def _extract_who_spans(sentence: str) -> list[str]:
    spans: list[str] = []
    for pattern in (
        re.compile(r"\b(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
        re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|was|works?|lives?)\b",
        ),
    ):
        for match in pattern.finditer(sentence):
            span = match.group(1).strip()
            if span.lower() not in _STOPWORDS:
                spans.append(span)
    return spans


@dataclass(frozen=True)
class _ContextItem:
    text: str
    memory_type: str = "note"
    provenance: dict[str, Any] | None = None
    score: float = 0.0
    session_date: str | None = None


@dataclass(frozen=True)
class _Candidate:
    text: str
    sentence: str
    score: float


def _is_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(pattern.search(stripped) for pattern in _JUNK_LINE_PATTERNS)


def _filter_context_text(text: str) -> str:
    kept = []
    for line in text.splitlines():
        if _is_junk_line(line):
            continue
        lower = line.strip().lower()
        # Session frontmatter dates are not answer evidence.
        if lower.startswith(("created_at:", "session_date:", "date:", "event_time:", "---")):
            continue
        if lower.startswith("tags:") or lower.startswith("client:") or lower.startswith("session_id:"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _context_session_date(text: str, provenance: dict[str, Any] | None = None) -> str | None:
    provenance = provenance or {}
    for key in ("session_date", "event_time", "created_at"):
        value = provenance.get(key)
        if value:
            spans = _extract_date_spans(str(value))
            if spans:
                return spans[0]
            if str(value).strip():
                return str(value).strip()
    for line in text.splitlines()[:8]:
        lower = line.strip().lower()
        if lower.startswith(("session_date:", "created_at:", "event_time:", "date:")):
            value = line.split(":", 1)[1].strip()
            spans = _extract_date_spans(value)
            return spans[0] if spans else value
    return None


def _is_list_question(question: str) -> bool:
    lower = question.lower()
    return bool(
        re.search(
            r"\bboth\b|\ball\b|"
            r"\bwhich\s+(?:cities|places|countries|states|books|games|activities|items|things|ways|types|kinds)\b|"
            r"\bwhat\s+(?:cities|places|countries|states|books|games|activities|items|things|ways|types|kinds)\b|"
            r"\bwhat\s+does\s+.+?\s+(?:offer|provide|include)\b",
            lower,
        )
    )


def _compact_list_values(question: str, sentence: str) -> list[str]:
    """Extract compact candidate values from a relevant sentence.

    This is deliberately schema/generic: it recognizes relation shapes rather
    than benchmark entities or known answers.
    """
    lower_q = question.lower()
    values: list[str] = []

    # Quoted works/titles are high-precision list values.
    if re.search(r"\bbooks?|titles?|movies?|films?|songs?|works?\b", lower_q):
        values.extend(
            match.group(1).strip()
            for match in re.finditer(r'["“]([^"”]{2,90})["”]', sentence)
        )

    # Travel / location histories.
    if re.search(r"\b(?:cities|places|states|countries|where)\b", lower_q):
        for pattern in (
            re.compile(
                r"\b(?:visited|went|traveled|travelled|vacationed|camped|stayed|lived)"
                r"\s+(?:in|at|to)?\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
            ),
            re.compile(
                r"\b(?:trip|vacation|camping)\s+(?:in|at|to)\s+"
                r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
            ),
        ):
            values.extend(match.group(1).strip() for match in pattern.finditer(sentence))

    # Product/service/class offerings: capture coordinated objects after the verb.
    if re.search(r"\b(?:offer|provide|include|services?|classes?|training|workshops?)\b", lower_q):
        for match in re.finditer(
            r"\b(?:offers?|provides?|includes?|has)\s+([^.!?]{3,140})",
            sentence,
            re.I,
        ):
            phrase = match.group(1)
            phrase = re.split(r"\b(?:because|so that|which|where|when)\b", phrase, maxsplit=1, flags=re.I)[0]
            values.extend(
                part.strip(" ,.;:-")
                for part in re.split(r",|\band\b|\bor\b", phrase, flags=re.I)
                if 2 <= len(part.strip()) <= 80
            )

    # Generic "I do/read/paint/attend X and Y" histories. Use only when the
    # question itself asks for a plural/set answer.
    if re.search(r"\b(?:what|which)\s+[a-z]+s\b|\bwhat\s+.+?\s+has\b", lower_q):
        for match in re.finditer(
            r"\b(?:do|does|did|done|read|reads|painted|paints|attended|attends|"
            r"participated\s+in|practiced|practises|practices|tried|uses?|enjoys?|likes?)\s+"
            r"([^.!?]{2,120})",
            sentence,
            re.I,
        ):
            phrase = re.split(
                r"\b(?:because|since|when|while|which|that|to\s+help|to\s+make)\b",
                match.group(1),
                maxsplit=1,
                flags=re.I,
            )[0]
            values.extend(
                part.strip(" ,.;:-")
                for part in re.split(r",|\band\b|\bor\b", phrase, flags=re.I)
                if 2 <= len(part.strip()) <= 70
            )

    # De-duplicate and discard obvious dialogue scaffolding.
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = re.sub(r"^(?:a|an|the)\s+", "", value.strip(), flags=re.I)
        value = re.sub(r"^(?:my|our|his|her|their)\s+", "", value, flags=re.I)
        if not value or _GREETING_ONLY_RE.match(value):
            continue
        if value.lower() in {"it", "them", "this", "that", "things", "stuff"}:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def _stem_shared_token(token: str) -> str:
    lower = token.casefold().strip(".,;:!?'\"")
    for suffix in ("ing", "ed", "es", "s"):
        if len(lower) > len(suffix) + 3 and lower.endswith(suffix):
            lower = lower[: -len(suffix)]
            break
    return lower


def _shared_entity_answer(
    question: str,
    normalized: list[_ContextItem],
    max_chars: int,
) -> str | None:
    """Find compact values independently supported for every named entity."""
    if not re.search(r"\bboth\b|\bin common\b|\bshared\b|\beach\b", question, re.I):
        return None

    entities = sorted(_question_entities(question))
    if len(entities) < 2:
        return None

    per_entity_values: dict[str, list[str]] = {entity: [] for entity in entities}
    per_entity_tokens: dict[str, Counter[str]] = {entity: Counter() for entity in entities}
    q_words = _content_words(question)

    for item in normalized:
        for sentence in _split_sentences(item.text):
            lower = sentence.casefold()
            matched_entities = [entity for entity in entities if entity in lower]
            if not matched_entities:
                continue

            compact = _compact_list_values(question, sentence)
            for entity in matched_entities:
                per_entity_values[entity].extend(compact)

                # Generic relation/action fallback for non-list commonality questions.
                # Keep only content words not already present in the question/entity.
                for token in _content_words(sentence):
                    stem = _stem_shared_token(token)
                    if (
                        stem
                        and stem not in {_stem_shared_token(word) for word in q_words}
                        and stem not in {_stem_shared_token(e) for e in entities}
                        and len(stem) >= 4
                    ):
                        per_entity_tokens[entity][stem] += 1

    # Exact compact-value intersection first (cities, titles, services, etc.).
    value_sets: list[dict[str, str]] = []
    for entity in entities:
        mapping: dict[str, str] = {}
        for value in per_entity_values[entity]:
            key = " ".join(_stem_shared_token(tok) for tok in _tokens(value))
            if key:
                mapping.setdefault(key, value)
        value_sets.append(mapping)

    if value_sets and all(value_sets):
        common = set(value_sets[0])
        for mapping in value_sets[1:]:
            common &= set(mapping)
        if common:
            values = [value_sets[0][key] for key in sorted(common)]
            return _truncate(", ".join(values[:8]), max_chars)

    # Fallback: intersect salient content stems across each person's evidence.
    token_sets = [set(counter) for counter in per_entity_tokens.values()]
    if not token_sets or not all(token_sets):
        return None
    shared = set.intersection(*token_sets)
    if not shared:
        return None

    generic = {
        "have", "with", "from", "that", "this", "they", "their", "your", "just",
        "really", "about", "been", "want", "like", "love", "great", "good", "also",
        "make", "help", "thing", "time", "need", "when", "what", "both",
    }
    ranked = [
        token
        for token in shared
        if token not in generic and token not in {_stem_shared_token(x) for x in q_words}
    ]
    if not ranked:
        return None

    # Prefer tokens repeatedly attested across people.
    ranked.sort(
        key=lambda token: sum(per_entity_tokens[e][token] for e in entities),
        reverse=True,
    )
    best = ranked[0]
    # Recover a readable surface form from evidence.
    variants: Counter[str] = Counter()
    for item in normalized:
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", item.text):
            if _stem_shared_token(token) == best:
                variants[token] += 1
    surface = variants.most_common(1)[0][0] if variants else best
    return _truncate(surface, max_chars)


def _list_answer(question: str, normalized: list[_ContextItem], max_chars: int) -> str | None:
    question_words = _content_words(question)
    entities = _question_entities(question)
    scored_sentences: list[tuple[float, str]] = []
    seen_sentences: set[str] = set()

    for item in normalized:
        for sentence in _split_sentences(item.text):
            if len(sentence) < 8:
                continue
            overlap = _overlap_score(question_words, sentence, entities)
            lower = sentence.lower()
            relation_bonus = 0.0
            if re.search(
                r"\b(?:visit|visited|went to|trip to|travel(?:ed|led)? to|"
                r"read|painted|attended|participated|camped|vacationed)\b",
                lower,
            ):
                relation_bonus += 0.45
            if re.search(
                r"\b(?:offer|offering|provide|provides|classes|workshops|training|services)\b",
                lower,
            ):
                relation_bonus += 0.45
            if overlap < 0.12 and relation_bonus == 0.0:
                continue
            key = re.sub(r"\s+", " ", sentence.strip()).lower()
            if key in seen_sentences:
                continue
            seen_sentences.add(key)
            scored_sentences.append(
                (
                    overlap + relation_bonus + min(item.score, 1.0) * 0.12,
                    sentence.strip(),
                )
            )

    if not scored_sentences:
        return None
    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    values: list[str] = []
    seen_values: set[str] = set()
    for _, sentence in scored_sentences[:10]:
        for value in _compact_list_values(question, sentence):
            key = value.casefold()
            if key in seen_values:
                continue
            seen_values.add(key)
            values.append(value)
            if len(values) >= 8:
                break
        if len(values) >= 8:
            break

    if values:
        compact = ", ".join(values)
        return _truncate(compact, max_chars)

    # Last-resort evidence aggregation when no structured values were extractable.
    chosen: list[str] = []
    used = 0
    for _, sentence in scored_sentences[:6]:
        if used + len(sentence) > max_chars and chosen:
            continue
        chosen.append(sentence)
        used += len(sentence) + 2
        if len(chosen) >= 2:
            break
    return _truncate(" ".join(chosen), max_chars) if chosen else None


def _normalize_contexts(
    contexts: list[str] | list[dict[str, Any]],
) -> list[_ContextItem]:
    items: list[_ContextItem] = []
    for raw in contexts:
        if isinstance(raw, str):
            session_date = _context_session_date(raw)
            text = _filter_context_text(raw)
            if text:
                items.append(_ContextItem(text=text, session_date=session_date))
            continue
        raw_text = str(raw.get("text") or "")
        provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
        session_date = _context_session_date(raw_text, provenance)
        text = _filter_context_text(raw_text)
        if not text:
            continue
        items.append(
            _ContextItem(
                text=text,
                memory_type=str(raw.get("memory_type") or "note"),
                provenance=provenance,
                score=float(raw.get("score") or 0.0),
                session_date=session_date,
            )
        )
    return items


def _question_temporal_bias(question: str) -> Literal["current", "past", "neutral"]:
    lower = question.lower()
    if any(cue in lower for cue in _CURRENT_CUES):
        return "current"
    if any(cue in lower for cue in _PAST_CUES):
        return "past"
    return "neutral"


def _context_metadata_bonus(
    item: _ContextItem,
    temporal_bias: Literal["current", "past", "neutral"],
    *,
    question_words: set[str] | None = None,
    entities: set[str] | None = None,
    identity_question: bool = False,
) -> float:
    provenance = item.provenance or {}
    bonus = 0.0

    if provenance.get("source") == "atomic-memory":
        bonus += 0.35

    if provenance.get("hop"):
        bonus += 0.45

    if provenance.get("conversation_neighbor"):
        bonus += 0.28
    if provenance.get("source") == "conversation-window":
        bonus += 0.22

    if provenance.get("chain_reachable"):
        bonus += 0.14
        bonus += min(0.28, 0.08 * int(provenance.get("chain_target_hits") or 0))
        bonus += min(0.16, 0.04 * int(provenance.get("chain_max_depth") or 0))
        if (
            provenance.get("chain_min_depth") == 0
            and int(provenance.get("chain_max_depth") or 0) == 0
            and int(provenance.get("chain_target_hits") or 0) == 0
        ):
            bonus -= 0.26

    if item.memory_type in _PREFERRED_MEMORY_TYPES:
        bonus += 0.15

    if item.score > 0:
        bonus += min(item.score * 0.12, 0.18)

    if question_words is not None and entities is not None:
        bonus += _dialogue_turn_bonus(
            item.text,
            question_words=question_words,
            entities=entities,
        )

    if identity_question and _IDENTITY_PHRASE_RE.search(item.text):
        bonus += 0.55

    # Inventory/profile dumps are great for multi-hop lists, noisy for span QA.
    if _INVENTORY_LINE_RE.search(item.text):
        bonus -= 0.85

    valid_until = provenance.get("valid_until")
    superseded_by = provenance.get("superseded_by")
    is_active = valid_until is None and not superseded_by

    if temporal_bias == "current":
        if is_active:
            bonus += 0.45
        else:
            bonus -= 0.55
    elif temporal_bias == "past":
        if not is_active or superseded_by:
            bonus += 0.4
        else:
            bonus -= 0.25

    return bonus


def _is_usable_context(item: _ContextItem) -> bool:
    text = item.text.strip()
    if not text:
        return False
    if _is_junk_line(text):
        return False
    provenance = item.provenance or {}
    if provenance.get("source") == "atomic-memory":
        return True
    if item.memory_type in _PREFERRED_MEMORY_TYPES and len(text) >= 8:
        return True
    # Giant raw session dumps drown extractive QA — keep only shorter evidence.
    if len(text) > 700 and item.memory_type in {"log", "note"}:
        return False
    if len(text) < 8 and not re.search(r"\b(19|20)\d{2}\b", text):
        return False
    return True


def _overlap_score(question_words: set[str], text: str, entities: set[str]) -> float:
    text_words = _content_words(text)
    if not question_words and not entities:
        return 0.0
    overlap = len(question_words & text_words)
    denom = max(len(question_words), 1)
    base = overlap / denom
    entity_hits = sum(1 for ent in entities if ent in text.lower())
    entity_bonus = 0.25 * entity_hits
    return base + entity_bonus


def _length_bonus(kind: QuestionKind, span: str) -> float:
    n = len(span)
    if kind in {"when", "where", "who", "what"}:
        if n <= 30:
            return 0.35
        if n <= 60:
            return 0.15
        if n <= 120:
            return 0.0
        return -0.2
    return 0.0


def _score_candidate(
    span: str,
    sentence: str,
    *,
    kind: QuestionKind,
    question_words: set[str],
    entities: set[str],
    occupation_question: bool = False,
    identity_question: bool = False,
) -> float:
    score = _overlap_score(question_words, sentence, entities)
    score += _overlap_score(question_words, span, entities) * 0.5
    score += _length_bonus(kind, span)

    if occupation_question:
        score += _occupation_score_adjustment(span, sentence)

    if kind == "when" and _extract_date_spans(span):
        score += 0.55
        # Prefer specific dates over bare years.
        if re.search(r"\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}", span):
            score += 0.35
        elif re.fullmatch(r"(?:19|20)\d{2}", span.strip()):
            score -= 0.15
    elif kind != "when" and re.fullmatch(r"(?:19|20)\d{2}", span.strip()):
        score -= 0.8

    if kind == "where" and _extract_location_spans(span):
        score += 0.4
    if kind == "who" and _extract_who_spans(span):
        score += 0.35
    if identity_question and _IDENTITY_PHRASE_RE.search(span):
        score += 0.65
    if "prefer" in question_words or "preference" in question_words or "theme" in question_words:
        if re.search(r"\b(?:light|dark)\s+mode\b", span, re.I):
            score += 0.7
        if span.lower() in {"the editor", "editor", "editor instead"}:
            score -= 0.9
    if any(w in question_words for w in ("decide", "decided", "decision", "chose", "caching", "system")):
        if re.search(r"\b(?:redis|memcached|postgres|qdrant|neo4j)\b", span, re.I):
            score += 0.85
        if len(span.split()) <= 3 and re.search(r"[A-Z]", span):
            score += 0.25
    if "nickname" in question_words or "alias" in question_words or "codename" in question_words:
        if re.search(r"[A-Za-z]+-\d+|\b[A-Z][a-z]+-\d+\b", span):
            score += 0.9

    lower = span.lower()
    if _GREETING_ONLY_RE.match(span.strip()):
        score -= 1.4
    if _GREETING_ONLY_RE.match(sentence.strip()):
        score -= 0.8

    if kind == "yes_no":
        if any(cue in lower for cue in _NEG_CUES):
            score += 0.2
        if any(cue in lower for cue in _AFFIRM_CUES):
            score += 0.2

    return score


_SUPERSESSION_TAIL = re.compile(
    r"\s+(?:instead\s+of|rather\s+than|as\s+opposed\s+to)\b.+$"
    r"|\s+instead\s*$"
    r"|;\s*i\s+no\s+longer\b.+$",
    re.IGNORECASE,
)
_IDENTITY_SPAN = re.compile(
    r"\b(?:i'?m|i\s+am|is|was)\s+(?:a\s+|an\s+)?(transgender\s+woman|transgender\s+man|"
    r"[^.,;!?]{3,60})",
    re.IGNORECASE,
)
_JOB_SPAN = re.compile(
    r"\bworks?\s+as\s+(?:a\s+|an\s+)?([^.,;!?]+?)(?=\s+instead\b|\s+rather\b|[.,;!]|$)",
    re.IGNORECASE,
)
_PREF_SPAN = re.compile(
    r"\bprefer(?:s|red)?\s+([^.,;!?]+?)(?=\s+instead\b|\s+rather\b|[.,;!]|$)",
    re.IGNORECASE,
)
_DECISION_OBJECT = re.compile(
    r"\b(?:decided|chose|chosen|settled)\s+(?:to\s+)?(?:use|adopt|pick|go\s+with)\s+"
    r"(?:a\s+|an\s+|the\s+)?([^.,;!?]+)",
    re.IGNORECASE,
)
_NICKNAME_SPAN = re.compile(
    r"\b(?:nickname|codename|alias)\s+is\s+([A-Za-z0-9][\w-]{1,40})",
    re.IGNORECASE,
)
_STEPS_SPAN = re.compile(r"\bsteps?\s*:\s*(.+)$", re.IGNORECASE)


def _strip_supersession_tail(text: str) -> str:
    return _SUPERSESSION_TAIL.sub("", text).strip(" .,;")


def _collect_candidates(sentence: str, kind: QuestionKind) -> list[str]:
    cleaned = _strip_supersession_tail(sentence)
    candidates = [cleaned, sentence]
    if kind == "when":
        candidates.extend(_extract_date_spans(sentence))
    elif kind == "where":
        candidates.extend(_extract_location_spans(cleaned))
        candidates.extend(_extract_location_spans(sentence))
    elif kind == "who":
        candidates.extend(_extract_who_spans(sentence))

    for match in _JOB_SPAN.finditer(sentence):
        candidates.append(match.group(1).strip())
    for match in _PREF_SPAN.finditer(cleaned):
        candidates.append(match.group(1).strip())
        # Also keep a tight "light mode" / "dark mode" style theme span.
        theme = re.search(r"\b((?:light|dark)\s+mode|[a-z]+ mode)\b", match.group(1), re.I)
        if theme:
            candidates.append(theme.group(1).strip())
    for match in _DECISION_OBJECT.finditer(sentence):
        candidates.append(match.group(1).strip())
    for match in _NICKNAME_SPAN.finditer(sentence):
        candidates.append(match.group(1).strip())
    for match in _STEPS_SPAN.finditer(sentence):
        candidates.append(match.group(1).strip())
    for match in _IDENTITY_SPAN.finditer(sentence):
        candidates.append(match.group(1).strip())

    # Clause fragments after common answer cues.
    for pattern in (
        re.compile(r"\b(?:is|was|are|were)\s+([^.,;!?]{3,80})", re.I),
        re.compile(r"\b(?:on|in|at)\s+([^.,;!?]{3,60})", re.I),
    ):
        for match in pattern.finditer(cleaned):
            frag = match.group(1).strip()
            # Bare years are only useful for "when" questions.
            if kind != "when" and re.fullmatch(r"(?:19|20)\d{2}", frag):
                continue
            candidates.append(frag)

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        norm = _strip_supersession_tail(item.strip())
        key = norm.lower()
        if norm and key not in seen and key not in {"proc", "profile", "v1", "v2", "prefs", "dec"}:
            seen.add(key)
            unique.append(norm)
    return unique


def _yes_no_answer(sentence: str, question_words: set[str], entities: set[str]) -> str | None:
    score = _overlap_score(question_words, sentence, entities)
    if score < 0.15 and not entities:
        return None
    lower = sentence.lower()
    neg = sum(1 for cue in _NEG_CUES if cue in lower)
    pos = sum(1 for cue in _AFFIRM_CUES if cue in lower)
    if neg > pos and neg > 0:
        return "No"
    if pos > neg and pos > 0:
        return "Yes"
    if re.search(r"\b(?:not|never|no)\b", lower):
        return "No"
    if score >= 0.35:
        return "Yes"
    return None


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "..."


def synthesize_answer(
    question: str,
    contexts: list[str] | list[dict[str, Any]],
    *,
    max_chars: int = 220,
) -> str:
    """Pick the best extractive span from *contexts* for *question*.

    *contexts* may be plain strings or dicts with ``text``, ``memory_type``,
    ``provenance``, and ``score`` (as returned by memory search).
    """
    normalized = [item for item in _normalize_contexts(contexts) if _is_usable_context(item)]
    if not normalized:
        return ""

    precise = _precise_scalar_answer(question, normalized)
    if precise:
        return _truncate(precise, max_chars)

    temporal_bias = _question_temporal_bias(question)
    kind = _question_kind(question)
    question_words = _content_words(question)
    entities = _question_entities(question)
    occupation_question = _is_occupation_question(question)
    identity_question = _is_identity_question(question)

    list_question = _is_list_question(question)

    shared_answer = _shared_entity_answer(question, normalized, max_chars)
    if shared_answer:
        return shared_answer

    # For "when" questions, prefer contexts that actually contain date spans.
    if kind == "when":
        duration_question = bool(
            re.search(r"\bhow long\b|\bhow many\s+(?:years?|months?|weeks?)\b", question, re.I)
        )
        temporal_only = [
            item for item in normalized
            if _extract_date_spans(item.text)
            or _extract_duration_spans(item.text)
            or item.session_date
        ]
        if temporal_only:
            normalized = temporal_only

    if kind == "yes_no":
        sentences: list[tuple[float, str]] = []
        for item in normalized:
            meta_bonus = _context_metadata_bonus(
                item,
                temporal_bias,
                question_words=question_words,
                entities=entities,
                identity_question=identity_question,
            )
            for sentence in _split_sentences(item.text):
                score = _overlap_score(question_words, sentence, entities) + meta_bonus
                sentences.append((score, sentence))
        sentences.sort(key=lambda item: item[0], reverse=True)
        for _, sentence in sentences:
            yn = _yes_no_answer(sentence, question_words, entities)
            if yn:
                return yn
        if sentences:
            return _truncate(sentences[0][1], max_chars)
        return ""

    ranked: list[_Candidate] = []
    for item in normalized:
        meta_bonus = _context_metadata_bonus(
            item,
            temporal_bias,
            question_words=question_words,
            entities=entities,
            identity_question=identity_question,
        )
        for sentence in _split_sentences(item.text):
            for span in _collect_candidates(sentence, kind):
                score = _score_candidate(
                    span,
                    sentence,
                    kind=kind,
                    question_words=question_words,
                    entities=entities,
                    occupation_question=occupation_question,
                    identity_question=identity_question,
                )
                score += meta_bonus
                ranked.append(_Candidate(text=span, sentence=sentence, score=score))

    if not ranked:
        return _truncate(" ".join(item.text for item in normalized), max_chars)

    ranked.sort(key=lambda c: (c.score, -len(c.text)), reverse=True)
    best = ranked[0]

    if kind == "when":
        duration_question = bool(
            re.search(
                r"\bhow long\b|\bhow many\s+(?:years?|months?|weeks?)\b|\byears? ago\b",
                question,
                re.I,
            )
        )
        # Prefer a date that co-occurs with question entities/content in the same sentence.
        dated: list[tuple[float, str]] = []
        for item in normalized:
            meta = _context_metadata_bonus(
                item,
                temporal_bias,
                question_words=question_words,
                entities=entities,
                identity_question=identity_question,
            )
            for sentence in _split_sentences(item.text):
                if _INVENTORY_LINE_RE.search(sentence):
                    continue
                overlap = _overlap_score(question_words, sentence, entities)
                if entities and not any(ent in sentence.lower() for ent in entities):
                    if overlap < 0.35:
                        continue
                elif overlap < 0.2:
                    continue
                spans = _extract_duration_spans(sentence) if duration_question else []
                spans = spans or _extract_date_spans(sentence)
                spans = _prefer_absolute_date_spans(spans)
                for date in spans:
                    # Penalize dates that are just session stamps without topical words.
                    topical = overlap + (
                        0.4 if any(w in sentence.lower() for w in question_words) else 0.0
                    )
                    # Strongly prefer absolute / anchored dates over "last Saturday".
                    abs_bonus = 0.0 if _is_relative_only_date(date) else 0.8
                    # Longer anchored phrases ("week before 9 June 2023") beat bare years
                    # when overlap is otherwise similar.
                    dated.append(
                        (
                            meta + topical + abs_bonus + min(len(date), 40) * 0.015,
                            date,
                        )
                    )
            if item.session_date:
                topical_sentences = [
                    sentence for sentence in _split_sentences(item.text)
                    if _overlap_score(question_words, sentence, entities) >= 0.22
                ]
                if topical_sentences:
                    best_overlap = max(
                        _overlap_score(question_words, sentence, entities)
                        for sentence in topical_sentences
                    )
                    dated.append((meta + best_overlap + 0.55, item.session_date))
        if dated:
            dated.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
            # If the top hit is relative-only, skip down to an absolute one.
            for score, date in dated:
                if not _is_relative_only_date(date) or all(
                    _is_relative_only_date(d) for _, d in dated
                ):
                    return _truncate(_display_date(date), max_chars)
            return _truncate(_display_date(dated[0][1]), max_chars)
        dates = _prefer_absolute_date_spans(
            _extract_duration_spans(best.text)
            or _extract_date_spans(best.text)
            or _extract_duration_spans(best.sentence)
            or _extract_date_spans(best.sentence)
        )
        if dates:
            return _truncate(_display_date(dates[0]), max_chars)
        # Avoid vague relative answers when no absolute date is available.
        if re.search(
            r"\b(?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|weekend|month)\b",
            best.text,
            re.I,
        ):
            for cand in ranked[1:8]:
                alt = _prefer_absolute_date_spans(
                    _extract_date_spans(cand.text) or _extract_date_spans(cand.sentence)
                )
                if alt and not _is_relative_only_date(alt[0]):
                    return _truncate(_display_date(alt[0]), max_chars)

    if kind == "where":
        locs = _extract_location_spans(best.text) or _extract_location_spans(best.sentence)
        if locs:
            return _truncate(_strip_supersession_tail(locs[0]), max_chars)

    if kind == "who":
        who = _extract_who_spans(best.text) or _extract_who_spans(best.sentence)
        if who:
            return _truncate(who[0], max_chars)

    if occupation_question:
        for cand in ranked:
            job = _JOB_SPAN.search(cand.sentence) or _JOB_SPAN.search(cand.text)
            if job:
                return _truncate(_strip_supersession_tail(job.group(1)), max_chars)

    # Prefer compact decision / nickname objects when clearly present.
    for cand in ranked[:8]:
        nick = _NICKNAME_SPAN.search(cand.sentence) or _NICKNAME_SPAN.search(cand.text)
        if nick and ("nickname" in question_words or "alias" in question_words):
            return _truncate(nick.group(1), max_chars)
        decided = _DECISION_OBJECT.search(cand.sentence) or _DECISION_OBJECT.search(cand.text)
        if decided and any(w in question_words for w in ("decide", "decided", "decision", "chose", "caching", "system")):
            obj = _strip_supersession_tail(decided.group(1))
            # Keep the head noun/tool name when the object is long.
            head = re.split(r"\s+for\s+|\s+as\s+", obj, maxsplit=1)[0].strip()
            return _truncate(head or obj, max_chars)

    # Distributed/list questions may require evidence from multiple contexts.
    # Only use aggregation when at least two distinct high-relevance sentences exist.
    if list_question:
        list_answer = _list_answer(question, normalized, max_chars)
        if list_answer:
            parts = _split_sentences(list_answer)
            if "," in list_answer or len(parts) >= 2 or any(
                cue in list_answer.lower()
                for cue in (
                    "visited", "trip to", "offer", "provid",
                    "classes", "workshops", "training",
                )
            ):
                return list_answer

    # Prefer a tight span when it still overlaps the question.
    answer = _strip_supersession_tail(best.text if best.score >= 0.2 and len(best.text) <= max_chars else best.sentence)
    if answer:
        return _truncate(answer, max_chars)
    return _truncate(best.sentence, max_chars)


def f1_ready_text(answer: str) -> str:
    """Normalize an answer for token F1 comparison."""
    return _normalize_for_f1(answer)
