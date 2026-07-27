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
    r"last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+year|this\s+month|this\s+week|recently|"
    r"last\s+weekend|this\s+weekend)\.?$",
    re.I,
)

_CAUSAL_HINTS = (
    "qwen",
    "llama",
    "mistral",
    "phi-",
    "gemma",
    "olmo",
    "smollm",
    "tinyllama",
)


def env_model_name() -> str:
    return os.environ.get("ULTIMATE_MEMORY_LOCAL_LLM", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def use_llm_from_env() -> bool:
    return os.environ.get("ULTIMATE_MEMORY_USE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _is_causal_model(model_name: str) -> bool:
    lower = model_name.lower()
    if "t5" in lower or "flan" in lower or "bart" in lower:
        return False
    return any(hint in lower for hint in _CAUSAL_HINTS)


def _clean_answer(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    # Drop leading "Answer:" echoes
    text = re.sub(r"^(?:answer|a)\s*:\s*", "", text, flags=re.I)
    # Causal models sometimes echo the question or add chat markers.
    text = re.sub(r"^(?:assistant|user|system)\s*:\s*", "", text, flags=re.I)
    if _UNANSWERABLE.match(text):
        return "I don't know"
    return text


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts)
    q_lower = question.lower()
    starts_with_wh = bool(re.match(r"^(?:what|which|who|where|when|how)\b", q_lower))
    yn_shape = bool(
        re.match(r"^(?:would|is|are|was|were|does|did|has|have|can|could)\b", q_lower)
        or re.search(r"\banswer yes or no\b", q_lower)
    )
    inferential = bool(
        re.search(r"\b(?:might|likely|would|could|potentially|suspected)\b", q_lower)
        or re.search(r"\b(?:around which|based on|underlying condition)\b", q_lower)
    )
    if yn_shape and not starts_with_wh:
        style = (
            "For this yes/no or would-question, answer like LoCoMo golds: "
            "'Likely yes', 'Likely no', 'Yes', 'No', or 'Yes; short reason'. "
            "If the question asks which option (book/author/team), name the option, not Yes/No. "
        )
    elif inferential and starts_with_wh:
        style = (
            "This is an inferential open-domain question. Return ONE short concrete "
            "entity (holiday, state, job title, technique, composer, org, nickname, "
            "condition, meat, game, park). Example: July 2 near US holiday → "
            "'Independence Day'. Do not write a sentence. "
        )
    elif starts_with_wh:
        style = (
            "Return the concrete answer span: a name, place, organization, technique, "
            "holiday, job title, or comma-separated list. Do NOT answer with only "
            "'Likely yes' or 'Likely no' unless the question is yes/no. "
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


def _looks_like_echo_or_meta(answer: str, question: str) -> bool:
    """True when the model echoed a question / session chitchat instead of answering."""
    a = answer.strip()
    if not a:
        return True
    if a.endswith("?"):
        return True
    lower = a.lower()
    if re.search(
        r"\b(?:any fun plans|catch up after|what's up with|tell me more|"
        r"hope you're|long time|that's gorgeous|got any other|"
        r"nice to hear|great to hear|what's been up)\b",
        lower,
    ):
        return True
    # Near-duplicate of the question.
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    a_tokens = set(re.findall(r"[a-z0-9]+", lower))
    if q_tokens and a_tokens and len(a_tokens & q_tokens) / max(len(a_tokens), 1) > 0.8:
        if len(a_tokens) <= len(q_tokens) + 2:
            return True
    return False


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

    if (
        not llm_answer
        or llm_answer.lower() == "i don't know"
        or _looks_like_echo_or_meta(llm_answer, question)
    ):
        return extractive or llm_answer

    upgraded = _prefer_absolute_date(llm_answer, contexts, question)
    # If LLM answer is long/noisy and extractive is a tight entity, prefer extractive.
    if extractive and len(extractive.split()) <= 6 and len(upgraded.split()) > 12:
        return extractive
    return upgraded


class LocalAnswerer:
    """Local answerer supporting seq2seq (Flan-T5) and causal instruct models (Qwen)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or env_model_name()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._causal = _is_causal_model(self.model_name)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM dependencies missing. Install with: uv sync --extra llm"
            ) from exc

        logger.info(
            "Loading local answerer model %s (%s)",
            self.model_name,
            "causal" if self._causal else "seq2seq",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._causal:
            # Prefer fp16/bf16 when CUDA is available; otherwise float32 CPU.
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def _generate(self, prompt: str, *, max_new_tokens: int = 48) -> str:
        self._ensure_loaded()
        if self._causal:
            return self._generate_causal(prompt, max_new_tokens=max_new_tokens)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generate_causal(self, prompt: str, *, max_new_tokens: int = 48) -> str:
        tok = self._tokenizer
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions from memory snippets only. "
                    "Reply with a short LoCoMo-style answer span or comma-separated list. "
                    "No preamble."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
        input_len = int(inputs["input_ids"].shape[-1])
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = outputs[0][input_len:]
        return tok.decode(gen, skip_special_tokens=True)

    def answer(self, question: str, contexts: list[str]) -> str:
        """Answer *question* from memory snippets in LoCoMo gold style."""
        trimmed = [c.strip() for c in contexts if c and c.strip()][:MAX_CONTEXTS]
        trimmed = [c[:MAX_CONTEXT_CHARS] for c in trimmed]
        if not trimmed:
            return ""

        prompt = _build_prompt(question, trimmed)
        raw = self._generate(prompt, max_new_tokens=64 if self._causal else 48)
        return hybrid_answer(question, trimmed, raw)

    def answer_list(self, question: str, contexts: list[str]) -> str:
        """Force comma-separated list answers for multi-hop inventory questions."""
        trimmed = [c.strip() for c in contexts if c and c.strip()][:24]
        trimmed = [c[:320] for c in trimmed]
        if not trimmed:
            return ""
        context_block = "\n".join(f"- {c}" for c in trimmed)
        prompt = (
            "From the memories, list every item that answers the question.\n"
            "Return ONLY a comma-separated list of short items (no sentences).\n"
            "Include all matching items found across memories.\n"
            'If none, reply "I don\'t know".\n\n'
            f"Memories:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Comma-separated list:"
        )
        raw = _clean_answer(self._generate(prompt, max_new_tokens=96 if self._causal else 64))
        if not raw or raw.lower() == "i don't know":
            return hybrid_answer(question, trimmed, raw)
        # Normalize separators.
        raw = raw.replace(" and ", ", ").replace("/", ", ").replace(";", ", ")
        parts = [p.strip(" .") for p in raw.split(",") if p.strip(" .")]
        # Drop duplicates, keep order.
        seen: set[str] = set()
        uniq: list[str] = []
        for part in parts:
            key = part.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(part)
        return ", ".join(uniq) if uniq else hybrid_answer(question, trimmed, raw)


@lru_cache(maxsize=2)
def get_local_answerer(model_name: str | None = None) -> LocalAnswerer:
    """Return a cached LocalAnswerer for *model_name*."""
    return LocalAnswerer(model_name=model_name or env_model_name())
