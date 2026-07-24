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
    re.compile(r"\b(?:in|on|during)\s+(?:the\s+)?(?:year\s+)?(19|20)\d{2}\b", re.I),
    re.compile(r"\b(19|20)\d{2}\b"),
    re.compile(r"\b(?:last|next)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I),
]

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


def _normalize_for_f1(text: str) -> str:
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = " ".join(tok for tok in no_punct.split() if tok not in _ARTICLES)
    return " ".join(no_articles.split())


def _tokens(text: str) -> list[str]:
    return _normalize_for_f1(text).split()


def tokenize_f1(prediction: str, ground_truth: str | list[str]) -> float:
    """Standard QA token F1 (SQuAD-style): lowercase, strip punctuation, max over golds."""
    golds = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
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
            key = span.lower()
            if key not in seen:
                seen.add(key)
                spans.append(span)
    return spans


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
    kept = [line for line in text.splitlines() if not _is_junk_line(line)]
    return "\n".join(kept).strip()


def _normalize_contexts(
    contexts: list[str] | list[dict[str, Any]],
) -> list[_ContextItem]:
    items: list[_ContextItem] = []
    for raw in contexts:
        if isinstance(raw, str):
            text = _filter_context_text(raw)
            if text:
                items.append(_ContextItem(text=text))
            continue
        text = _filter_context_text(str(raw.get("text") or ""))
        if not text:
            continue
        items.append(
            _ContextItem(
                text=text,
                memory_type=str(raw.get("memory_type") or "note"),
                provenance=raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {},
                score=float(raw.get("score") or 0.0),
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


def _context_metadata_bonus(item: _ContextItem, temporal_bias: Literal["current", "past", "neutral"]) -> float:
    provenance = item.provenance or {}
    bonus = 0.0

    if provenance.get("source") == "atomic-memory":
        bonus += 0.35

    if provenance.get("hop"):
        bonus += 0.45

    if item.memory_type in _PREFERRED_MEMORY_TYPES:
        bonus += 0.15

    if item.score > 0:
        bonus += min(item.score * 0.12, 0.18)

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
) -> float:
    score = _overlap_score(question_words, sentence, entities)
    score += _overlap_score(question_words, span, entities) * 0.5
    score += _length_bonus(kind, span)

    if occupation_question:
        score += _occupation_score_adjustment(span, sentence)

    if kind == "when" and _extract_date_spans(span):
        score += 0.55
    if kind == "where" and _extract_location_spans(span):
        score += 0.4
    if kind == "who" and _extract_who_spans(span):
        score += 0.35

    lower = span.lower()
    if kind == "yes_no":
        if any(cue in lower for cue in _NEG_CUES):
            score += 0.2
        if any(cue in lower for cue in _AFFIRM_CUES):
            score += 0.2

    return score


def _collect_candidates(sentence: str, kind: QuestionKind) -> list[str]:
    candidates = [sentence]
    if kind == "when":
        candidates.extend(_extract_date_spans(sentence))
    elif kind == "where":
        candidates.extend(_extract_location_spans(sentence))
    elif kind == "who":
        candidates.extend(_extract_who_spans(sentence))

    # Clause fragments after common answer cues.
    for pattern in (
        re.compile(r"\b(?:is|was|are|were)\s+([^.,;!?]{3,80})", re.I),
        re.compile(r"\b(?:on|in|at)\s+([^.,;!?]{3,60})", re.I),
    ):
        for match in pattern.finditer(sentence):
            candidates.append(match.group(1).strip())

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        norm = item.strip()
        key = norm.lower()
        if norm and key not in seen:
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

    temporal_bias = _question_temporal_bias(question)
    kind = _question_kind(question)
    question_words = _content_words(question)
    entities = _question_entities(question)
    occupation_question = _is_occupation_question(question)

    if kind == "yes_no":
        sentences: list[tuple[float, str]] = []
        for item in normalized:
            meta_bonus = _context_metadata_bonus(item, temporal_bias)
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
        meta_bonus = _context_metadata_bonus(item, temporal_bias)
        for sentence in _split_sentences(item.text):
            for span in _collect_candidates(sentence, kind):
                score = _score_candidate(
                    span,
                    sentence,
                    kind=kind,
                    question_words=question_words,
                    entities=entities,
                    occupation_question=occupation_question,
                )
                score += meta_bonus
                ranked.append(_Candidate(text=span, sentence=sentence, score=score))

    if not ranked:
        return _truncate(" ".join(item.text for item in normalized), max_chars)

    ranked.sort(key=lambda c: (c.score, -len(c.text)), reverse=True)
    best = ranked[0]

    if kind == "when":
        dates = _extract_date_spans(best.text) or _extract_date_spans(best.sentence)
        if dates:
            return _truncate(dates[0], max_chars)

    if kind == "where":
        locs = _extract_location_spans(best.text) or _extract_location_spans(best.sentence)
        if locs:
            return _truncate(locs[0], max_chars)

    if kind == "who":
        who = _extract_who_spans(best.text) or _extract_who_spans(best.sentence)
        if who:
            return _truncate(who[0], max_chars)

    # Prefer a tight span when it still overlaps the question.
    if best.score >= 0.2 and len(best.text) <= max_chars:
        return best.text.strip()

    return _truncate(best.sentence, max_chars)


def f1_ready_text(answer: str) -> str:
    """Normalize an answer for token F1 comparison."""
    return _normalize_for_f1(answer)
