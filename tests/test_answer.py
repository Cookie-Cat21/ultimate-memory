"""Tests for extractive answer synthesis and token F1."""

from __future__ import annotations

from pathlib import Path

from ultimate_memory.answer import synthesize_answer, tokenize_f1
from ultimate_memory.config import (
    BasicMemoryConfig,
    DashboardConfig,
    Neo4jConfig,
    PathsConfig,
    QdrantConfig,
    RetrievalConfig,
    Settings,
)
from ultimate_memory.router import MemoryRouter


def make_settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Redis.md").write_text("# Redis\n\nCache notes.", encoding="utf-8")
    private = tmp_path / "private"
    return Settings(
        paths=PathsConfig(repo_root=tmp_path, basic_memory_vault=vault, private_store=private),
        retrieval=RetrievalConfig(
            collection_name="test",
            embedding_model="BAAI/bge-small-en-v1.5",
            chunk_chars=300,
            chunk_overlap=40,
            default_limit=5,
            bootstrap_token_budget_chars=2000,
        ),
        qdrant=QdrantConfig(url="http://127.0.0.1:1"),
        neo4j=Neo4jConfig(uri="bolt://127.0.0.1:1", user="neo4j", password="password"),
        basic_memory=BasicMemoryConfig(project="main", cli="basic-memory-missing"),
        dashboard=DashboardConfig(host="127.0.0.1", port=8787),
    )


class TestTokenizeF1:
    def test_exact_match(self):
        assert tokenize_f1("Paris", "Paris") == 1.0

    def test_case_and_punctuation_insensitive(self):
        assert tokenize_f1("Paris.", "paris") == 1.0

    def test_partial_overlap(self):
        f1 = tokenize_f1("May 2023", "in May 2023")
        assert 0.5 < f1 < 1.0

    def test_max_over_multiple_golds(self):
        assert tokenize_f1("Austin", ["Boston", "Austin"]) == 1.0

    def test_empty_prediction(self):
        assert tokenize_f1("", "something") == 0.0


class TestSynthesizeAnswer:
    WHEN_CONTEXTS = [
        "Maria talked about her career. She moved to Seattle in May 2023 for a new job.",
        "Before that she lived in Portland for two years.",
    ]
    WHERE_CONTEXTS = [
        "Alex grew up in Chicago. Last summer Alex visited Denver for a conference.",
        "Alex now lives in Austin, Texas.",
    ]
    YES_NO_CONTEXTS = [
        "Sam loves hiking on weekends and goes every Saturday.",
        "Sam does not enjoy crowded beaches.",
    ]

    def test_when_question_picks_date_span(self):
        answer = synthesize_answer(
            "When did Maria move to Seattle?",
            self.WHEN_CONTEXTS,
        )
        assert "2023" in answer
        f1 = tokenize_f1(answer, "May 2023")
        assert f1 >= 0.5

    def test_where_question_picks_location(self):
        answer = synthesize_answer(
            "Where does Alex live now?",
            self.WHERE_CONTEXTS,
        )
        assert "Austin" in answer
        f1 = tokenize_f1(answer, "Austin")
        assert f1 >= 0.5

    def test_yes_no_question(self):
        answer = synthesize_answer(
            "Does Sam enjoy hiking?",
            self.YES_NO_CONTEXTS,
        )
        assert answer in {"Yes", "No"}
        assert tokenize_f1(answer, "Yes") == 1.0

    def test_yes_no_negative(self):
        answer = synthesize_answer(
            "Does Sam enjoy crowded beaches?",
            self.YES_NO_CONTEXTS,
        )
        assert answer == "No"

    def test_fallback_truncates_best_sentence(self):
        contexts = ["Unrelated intro. The capital of France is Paris and it is famous."]
        answer = synthesize_answer("What is the capital of France?", contexts, max_chars=40)
        assert "Paris" in answer
        assert len(answer) <= 43

    def test_empty_contexts(self):
        assert synthesize_answer("Where?", []) == ""

    def test_prefers_active_atom_over_stale_note_for_current_question(self):
        contexts = [
            {
                "text": "Carol lives in New York.",
                "memory_type": "note",
                "score": 0.9,
                "provenance": {"source": "sqlite-fts"},
            },
            {
                "text": "Carol moved to Austin in 2024 for a new role.",
                "memory_type": "fact",
                "score": 0.7,
                "provenance": {
                    "source": "atomic-memory",
                    "valid_until": None,
                    "superseded_by": None,
                },
            },
        ]
        answer = synthesize_answer("Where does Carol live now?", contexts)
        assert "Austin" in answer
        assert "New York" not in answer

    def test_skips_markdown_relation_junk(self):
        contexts = [
            "- mentions [[Caroline]]",
            {
                "text": "Caroline works remotely from Denver.",
                "memory_type": "fact",
                "score": 0.8,
                "provenance": {"source": "atomic-memory", "valid_until": None},
            },
        ]
        answer = synthesize_answer("Where does Caroline work from?", contexts)
        assert "Denver" in answer
        assert "mentions" not in answer.lower()
        assert "[[" not in answer

    def test_boosts_superseded_for_past_tense_question(self):
        contexts = [
            {
                "text": "Carol lives in Austin.",
                "memory_type": "fact",
                "score": 0.85,
                "provenance": {
                    "source": "atomic-memory",
                    "valid_until": None,
                    "superseded_by": None,
                },
            },
            {
                "text": "Carol lived in New York before moving.",
                "memory_type": "fact",
                "score": 0.6,
                "provenance": {
                    "source": "atomic-memory",
                    "valid_until": "2024-01-01T00:00:00+00:00",
                    "superseded_by": "atom:other",
                },
            },
        ]
        answer = synthesize_answer("Where did Carol live before?", contexts)
        assert "New York" in answer

    def test_filters_reflection_markdown_sections(self):
        junk_note = "\n".join(
            [
                "# Memory Reflection",
                "",
                "## Open Questions",
                "- Will Carol relocate again?",
                "",
                "## Source Refs",
                "- session:abc",
                "",
                "## Relations",
                "- mentions [[Caroline]]",
                "- relates_to [[Shared Goal]]",
                "- from_project [[demo]]",
            ]
        )
        contexts = [
            junk_note,
            {
                "text": "Carol moved to Austin in 2024.",
                "memory_type": "fact",
                "score": 0.75,
                "provenance": {"source": "atomic-memory", "valid_until": None},
            },
        ]
        answer = synthesize_answer("Where does Carol live now?", contexts)
        assert "Austin" in answer
        assert "mentions" not in answer.lower()
        assert "Open Questions" not in answer


class TestMemoryRouterAnswer:
    def test_answer_wraps_search_and_synthesis(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.ingest_log(
            client="locomo",
            session_id="s1",
            transcript_or_path=(
                "Maria: I moved to Seattle in May 2023 for my new job.\n"
                "Maria: Before that I lived in Portland."
            ),
        )

        result = router.answer("When did Maria move to Seattle?", limit=5)

        assert "2023" in result["answer"]
        assert result["f1_text"]
        assert result["contexts_used"]
        assert result["search"]["results"]
        f1 = tokenize_f1(result["answer"], "May 2023")
        assert f1 >= 0.5
