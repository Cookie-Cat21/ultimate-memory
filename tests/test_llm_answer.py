"""Unit tests for optional local HuggingFace answerer (mocked, no downloads)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ultimate_memory.llm_answer import (
    LocalAnswerer,
    _clean_answer,
    get_local_answerer,
)


class TestCleanAnswer:
    def test_normalizes_i_dont_know(self):
        assert _clean_answer("I don't know") == "I don't know"
        assert _clean_answer("  unknown  ") == "I don't know"

    def test_preserves_short_span(self):
        assert _clean_answer("  May 2023  ") == "May 2023"


class TestLocalAnswererMocked:
    def setup_method(self) -> None:
        get_local_answerer.cache_clear()

    def test_answer_uses_model_generate(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "fake"}
        mock_tokenizer.decode.return_value = "Seattle"

        mock_model = MagicMock()
        mock_model.generate.return_value = [0]

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict("sys.modules", {"torch": mock_torch, "transformers": MagicMock()}),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "transformers.AutoModelForSeq2SeqLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            answerer = LocalAnswerer(model_name="test/model")
            result = answerer.answer(
                "Where did Maria move?",
                ["Maria moved to Seattle in May 2023."],
            )

        assert result == "Seattle"
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()

    def test_empty_contexts_returns_empty(self):
        answerer = LocalAnswerer(model_name="test/model")
        assert answerer.answer("Where?", []) == ""

    def test_missing_deps_raises(self):
        answerer = LocalAnswerer(model_name="test/model")
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with pytest.raises(RuntimeError, match="dependencies missing"):
                answerer.answer("Where?", ["context"])
