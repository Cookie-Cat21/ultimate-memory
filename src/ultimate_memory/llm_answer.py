"""Optional local HuggingFace answerer for LoCoMo-style short answers."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/flan-t5-base"
MAX_CONTEXTS = 6
MAX_CONTEXT_CHARS = 400

_UNANSWERABLE = re.compile(
    r"^(?:i\s+don'?t\s+know|unknown|n/?a|none|not\s+(?:mentioned|stated|found))\.?$",
    re.I,
)


def env_model_name() -> str:
    return os.environ.get("ULTIMATE_MEMORY_LOCAL_LLM", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def use_llm_from_env() -> bool:
    return os.environ.get("ULTIMATE_MEMORY_USE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _clean_answer(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if _UNANSWERABLE.match(text):
        return "I don't know"
    return text


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts)
    return (
        "Answer the question using only the memory snippets. "
        "Reply with a short phrase or span (names, dates, places). "
        'If the snippets do not contain the answer, reply "I don\'t know".\n\n'
        f"Memories:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class LocalAnswerer:
    """Seq2seq local answerer (default: google/flan-t5-base)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or env_model_name()
        self._model = None
        self._tokenizer = None

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
            max_length=512,
        )
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=2,
                early_stopping=True,
            )
        raw = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _clean_answer(raw)


@lru_cache(maxsize=1)
def get_local_answerer(model_name: str | None = None) -> LocalAnswerer:
    """Return a cached singleton LocalAnswerer."""
    return LocalAnswerer(model_name=model_name or env_model_name())
