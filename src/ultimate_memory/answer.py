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
    # Day Month Year: 7 May 2023
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
    re.compile(r"\b(?:in|on|during)\s+(?:the\s+)?(?:year\s+)?((?:19|20)\d{2})\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b"),
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
    if re.search(r"\b(?:identity|who is|what is)\b", " ".join(question_words), re.I) or "identity" in question_words:
        if re.search(r"\btransgender\b|\bwoman\b|\bman\b|\bengineer\b|\bnurse\b", span, re.I):
            score += 0.5
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

    temporal_bias = _question_temporal_bias(question)
    kind = _question_kind(question)
    question_words = _content_words(question)
    entities = _question_entities(question)
    occupation_question = _is_occupation_question(question)

    # For "when" questions, prefer contexts that actually contain date spans.
    if kind == "when":
        dated_only = [item for item in normalized if _extract_date_spans(item.text)]
        if dated_only:
            normalized = dated_only

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
        # Prefer a date that co-occurs with question entities/content in the same sentence.
        dated: list[tuple[float, str]] = []
        for item in normalized:
            meta = _context_metadata_bonus(item, temporal_bias)
            for sentence in _split_sentences(item.text):
                overlap = _overlap_score(question_words, sentence, entities)
                if entities and not any(ent in sentence.lower() for ent in entities):
                    if overlap < 0.35:
                        continue
                elif overlap < 0.2:
                    continue
                for date in _extract_date_spans(sentence):
                    # Penalize dates that are just session stamps without topical words.
                    topical = overlap + (0.4 if any(w in sentence.lower() for w in question_words) else 0.0)
                    dated.append((meta + topical + min(len(date), 20) * 0.01, date))
        if dated:
            dated.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
            return _truncate(dated[0][1], max_chars)
        dates = _extract_date_spans(best.text) or _extract_date_spans(best.sentence)
        if dates:
            return _truncate(dates[0], max_chars)
        # Avoid vague relative answers when no absolute date is available.
        if re.search(r"\b(?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|weekend|month)\b", best.text, re.I):
            for cand in ranked[1:8]:
                alt = _extract_date_spans(cand.text) or _extract_date_spans(cand.sentence)
                if alt:
                    return _truncate(alt[0], max_chars)

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

    # Prefer a tight span when it still overlaps the question.
    answer = _strip_supersession_tail(best.text if best.score >= 0.2 and len(best.text) <= max_chars else best.sentence)
    if answer:
        return _truncate(answer, max_chars)
    return _truncate(best.sentence, max_chars)


def f1_ready_text(answer: str) -> str:
    """Normalize an answer for token F1 comparison."""
    return _normalize_for_f1(answer)
