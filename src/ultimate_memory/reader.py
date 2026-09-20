"""Optional benchmark-agnostic extractive QA reader.

The memory system remains usable without this module's optional dependencies.
When enabled, a standard extractive QA model selects answer spans from retrieved
memory evidence. No benchmark categories, dataset entities, or gold annotations
are provided to the reader.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from .answer import synthesize_answer

DEFAULT_READER_MODEL = "distilbert-base-cased-distilled-squad"
MAX_READER_CONTEXTS = 8
MAX_CONTEXT_CHARS = 1800

_YES_NO = re.compile(
    r"^(?:is|are|was|were|do|does|did|has|have|had|can|could|would|will|should|may|might)\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\bwhen\b|\bwhat\s+(?:date|year|month|day)\b|"
    r"\bhow\s+long\b|\bhow\s+many\s+(?:years?|months?|weeks?|days?)\b",
    re.I,
)
_DISTRIBUTED = re.compile(
    r"\bboth\b|\ball\b|"
    r"\b(?:what|which)\s+(?!(?:does|has|is|was|this)\b)[a-z]+s\b"
    r".*\b(?:has|have|did|does|are|were)\b|"
    r"\bwhere\s+has\s+.+?\s+(?:camped|traveled|travelled|visited|stayed|lived)\b|"
    r"\bwhat\s+does\s+.+?\s+(?:offer|provide|include|do\s+to)\b",
    re.I,
)


def reader_model_name() -> str:
    return os.environ.get("ULTIMATE_MEMORY_READER_MODEL", DEFAULT_READER_MODEL).strip() or DEFAULT_READER_MODEL


class ExtractiveReader:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or reader_model_name()
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Reader dependencies missing. Install with: uv sync --extra llm"
            ) from exc
        self._pipeline = pipeline(
            "question-answering",
            model=self.model_name,
            tokenizer=self.model_name,
            device=-1,
        )

    @staticmethod
    def _text(item: str | dict[str, Any]) -> str:
        if isinstance(item, str):
            return item.strip()
        return str(item.get("text") or "").strip()

    def answer(self, question: str, contexts: list[str] | list[dict[str, Any]]) -> str:
        """Return the highest-confidence answer span from top retrieved evidence."""
        trimmed = [
            self._text(item)[:MAX_CONTEXT_CHARS]
            for item in contexts[:MAX_READER_CONTEXTS]
            if self._text(item)
        ]
        if not trimmed:
            return ""

        # Span readers cannot faithfully synthesize yes/no, temporal metadata,
        # or answers distributed across multiple memories. Route those shapes
        # through the deterministic structured reasoner first.
        if (
            _YES_NO.match(question.strip())
            or _TEMPORAL.search(question)
            or _DISTRIBUTED.search(question)
        ):
            structured = synthesize_answer(question, contexts)
            if structured:
                return structured

        self._ensure_loaded()
        payloads = [{"question": question, "context": text} for text in trimmed]
        outputs = self._pipeline(payloads, batch_size=min(8, len(payloads)))
        if isinstance(outputs, dict):
            outputs = [outputs]

        best_answer = ""
        best_score = -1.0
        for output in outputs:
            answer = str(output.get("answer") or "").strip()
            score = float(output.get("score") or 0.0)
            if not answer:
                continue
            if score > best_score:
                best_score = score
                best_answer = answer

        if not best_answer or best_score < 0.015:
            return synthesize_answer(question, contexts)
        return best_answer


@lru_cache(maxsize=2)
def get_extractive_reader(model_name: str | None = None) -> ExtractiveReader:
    return ExtractiveReader(model_name=model_name or reader_model_name())
