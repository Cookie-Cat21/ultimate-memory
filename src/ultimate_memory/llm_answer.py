"""Optional local HuggingFace answerer for LoCoMo-style short answers."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from .answer import _extract_date_spans, synthesize_answer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/flan-t5-base"
MAX_CONTEXTS = 18
MAX_CONTEXT_CHARS = 420

_UNANSWERABLE = re.compile(
    r"^(?:i\s+don'?t\s+know|unknown|n/?a|none|not\s+(?:mentioned|stated|found))\.?$",
    re.I,
)
_RELATIVE = re.compile(
    r"^(?:yesterday|today|tomorrow|last\s+year|last\s+month|last\s+week|"
    r"this\s+year|this\s+month|this\s+week|recently)\.?$",
    re.I,
)


def env_model_name() -> str:
    return os.environ.get("ULTIMATE_MEMORY_LOCAL_LLM", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def use_llm_from_env() -> bool:
    return os.environ.get("ULTIMATE_MEMORY_USE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _clean_answer(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    # Drop leading "Answer:" echoes
    text = re.sub(r"^(?:answer|a)\s*:\s*", "", text, flags=re.I)
    if _UNANSWERABLE.match(text):
        return "I don't know"
    return text


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts)
    q_lower = question.lower()
    if re.search(r"\bwould\b|\bmight\b|\blikely\b|\bconsidered\b", q_lower):
        style = (
            "For this inferential question, answer like LoCoMo golds: "
            "'Likely yes', 'Likely no', 'Yes; reason', or a short inferred label/list. "
        )
    elif re.search(r"\b(?:what|which)\b.+\b(?:has|have)\b", q_lower):
        style = "If multiple items fit, return a comma-separated list. "
    else:
        style = (
            "Return a SHORT answer phrase that matches the question "
            "(a name, date like '7 May 2023', place, job, or yes/no). "
        )
    return (
        "You are a memory QA system. Use ONLY the memory snippets below.\n"
        f"{style}"
        "Prefer absolute dates over words like yesterday/last year when both appear.\n"
        'If the snippets do not contain the answer, reply "I don\'t know".\n\n'
        f"Memories:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Short answer:"
    )


def _prefer_absolute_date(llm_answer: str, contexts: list[str], question: str) -> str:
    """If the LLM returns a relative date, prefer an absolute date span from contexts."""
    from .answer import _is_relative_only_date, _prefer_absolute_date_spans

    if not (
        _RELATIVE.match(llm_answer.strip())
        or _is_relative_only_date(llm_answer.strip())
        or re.search(r"\bwhen\b|\bhow long\b|\bwhat\s+(?:year|date|month|day)\b", question, re.I)
    ):
        return llm_answer
    extractive = synthesize_answer(
        question,
        [{"text": c, "memory_type": "fact", "provenance": {"source": "atomic-memory"}, "score": 1.0} for c in contexts],
    )
    if extractive and not _is_relative_only_date(extractive):
        # Prefer extractive temporal spans (incl. "4 years", "week before ...").
        if re.search(
            r"\b(?:19|20)\d{2}\b|\b\d+\s+years?\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            extractive,
            re.I,
        ):
            return extractive
    dates = _prefer_absolute_date_spans(
        _extract_date_spans(extractive) or _extract_date_spans(" ".join(contexts))
    )
    if dates:
        # Prefer day-month-year / anchored phrases over bare year when available.
        dates = sorted(dates, key=lambda d: (not _is_relative_only_date(d), len(d)), reverse=True)
        return dates[0]
    return llm_answer


def hybrid_answer(question: str, contexts: list[str], llm_answer: str) -> str:
    """Blend local LLM output with extractive spans for LoCoMo F1."""
    llm_answer = _clean_answer(llm_answer or "")
    extractive = synthesize_answer(
        question,
        [
            {
                "text": c,
                "memory_type": "fact",
                "provenance": {"source": "atomic-memory"},
                "score": 1.0,
            }
            for c in contexts
        ],
    )
    q_lower = question.lower()

    # Temporal: absolute dates win.
    if re.search(r"\bwhen\b|\bwhat\s+(?:year|date|month|day)\b", q_lower):
        ext_dates = _extract_date_spans(extractive)
        if ext_dates:
            return sorted(ext_dates, key=len, reverse=True)[0]
        if llm_answer and llm_answer.lower() != "i don't know":
            return _prefer_absolute_date(llm_answer, contexts, question)

    if not llm_answer or llm_answer.lower() == "i don't know":
        return extractive

    upgraded = _prefer_absolute_date(llm_answer, contexts, question)
    # If LLM answer is long/noisy and extractive is a tight entity, prefer extractive.
    if extractive and len(extractive.split()) <= 6 and len(upgraded.split()) > 12:
        return extractive
    return upgraded


class LocalAnswerer:
    """Seq2seq local answerer (default: google/flan-t5-base)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or env_model_name()
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM dependencies missing. Install with: uv sync --extra llm"
            ) from exc

        logger.info("Loading local answerer model %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def answer(self, question: str, contexts: list[str]) -> str:
        """Answer *question* from memory snippets in LoCoMo gold style."""
        trimmed = [c.strip() for c in contexts if c and c.strip()][:MAX_CONTEXTS]
        trimmed = [c[:MAX_CONTEXT_CHARS] for c in trimmed]
        if not trimmed:
            return ""

        self._ensure_loaded()
        prompt = _build_prompt(question, trimmed)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=48,
                num_beams=4,
                early_stopping=True,
            )
        raw = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return hybrid_answer(question, trimmed, raw)


@lru_cache(maxsize=1)
def get_local_answerer(model_name: str | None = None) -> LocalAnswerer:
    """Return a cached singleton LocalAnswerer."""
    return LocalAnswerer(model_name=model_name or env_model_name())
