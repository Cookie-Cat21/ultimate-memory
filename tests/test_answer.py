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


def test_when_can_use_relevant_session_frontmatter_date():
    contexts = [
        {
            "text": "---\nsession_date: 4 February, 2023\n---\n[D4:3] Jon: My group is performing at the festival this month.",
            "memory_type": "log",
            "score": 0.9,
        }
    ]
    answer = synthesize_answer("When is Jon's group performing at a festival?", contexts)
    assert "February" in answer and "2023" in answer


def test_relative_multi_year_date_is_extractable():
    contexts = [{"text": "Gina: I got my tattoo a few years ago.", "score": 1.0}]
    assert synthesize_answer("When did Gina get her tattoo?", contexts).lower() == "a few years ago"


def test_list_question_combines_distributed_evidence():
    contexts = [
        {"text": "Jon: I visited Paris last winter.", "score": 0.9},
        {"text": "Jon: I took a trip to Rome this summer.", "score": 0.8},
    ]
    answer = synthesize_answer("Which cities has Jon visited?", contexts)
    assert "Paris" in answer
    assert "Rome" in answer


def test_word_number_duration_is_preferred():
    contexts = [
        {"text": "Caroline: I've had this group of friends for four years now.", "score": 0.9},
        {"text": "---\nsession_date: 7 May 2023\n---\nCaroline: We met up recently.", "score": 0.5},
    ]
    answer = synthesize_answer("How long has Caroline had this group of friends?", contexts)
    assert "four years" in answer.lower()


def test_greeting_only_candidate_is_penalized():
    contexts = [
        {"text": "Gina: Wow!", "score": 1.0},
        {"text": "Gina: Dance feels magical to me.", "score": 0.8},
    ]
    answer = synthesize_answer("How does Gina describe the feeling that dance brings?", contexts)
    assert "magical" in answer.lower()


def test_answer_expands_to_adjacent_conversation_turn(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    transcript = """---
session_date: 10 May 2023
---
[D1:1] Melanie: What pet do you have?
[D1:2] Caroline: I have a guinea pig named Clover.
[D1:3] Melanie: That sounds adorable.
"""
    router.ingest_log(
        client="test",
        session_id="adjacency",
        transcript_or_path=transcript,
        project_path="/project",
    )
    result = router.answer("What pet does Caroline have?", project_path="/project", limit=6)
    assert any(
        (item.get("provenance") or {}).get("conversation_neighbor")
        for item in result["search"].get("results", [])
    ) is False  # neighbors are answer-context expansion, not base search output
    assert "guinea pig" in " ".join(result["contexts_used"]).lower()


def test_list_synthesis_returns_compact_locations():
    contexts = [
        {"text": "Jon: I visited Paris last winter.", "score": 0.9},
        {"text": "Jon: I traveled to Rome this summer.", "score": 0.8},
    ]
    answer = synthesize_answer("Which cities has Jon visited?", contexts)
    assert "Paris" in answer and "Rome" in answer
    assert len(answer) < 80


def test_list_synthesis_extracts_quoted_titles():
    contexts = [
        {"text": 'Alex: I read "Dune" last month.', "score": 0.9},
        {"text": 'Alex: I also read "The Hobbit" this year.', "score": 0.8},
    ]
    answer = synthesize_answer("What books has Alex read?", contexts)
    assert "Dune" in answer and "The Hobbit" in answer
    assert len(answer) < 80


def test_ingestion_indexes_question_response_pair(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    transcript = """---
session_date: 10 May 2023
---
[D1:1] Melanie: What pet do you have?
[D1:2] Caroline: I have a guinea pig named Clover.
"""
    router.ingest_log(
        client="test",
        session_id="pair-index",
        transcript_or_path=transcript,
        project_path="/project",
    )
    result = router.search("pet Caroline", project_path="/project", limit=10)
    pair_hits = [
        item for item in result["results"]
        if "What pet do you have?" in item["text"] and "guinea pig" in item["text"]
    ]
    assert pair_hits


def test_shared_city_intersection_across_people():
    contexts = [
        {"text": "Jean: I visited Rome last spring.", "score": 0.9},
        {"text": "Jean: I also visited Paris.", "score": 0.8},
        {"text": "John: I traveled to Rome last year.", "score": 0.9},
        {"text": "John: I visited Berlin too.", "score": 0.8},
    ]
    answer = synthesize_answer("Which city have both Jean and John visited?", contexts)
    assert "Rome" in answer
    assert "Paris" not in answer
    assert "Berlin" not in answer


def test_shared_activity_intersection_uses_entity_evidence():
    contexts = [
        {"text": "Jon: I dance whenever I need to destress.", "score": 0.9},
        {"text": "Gina: Dancing helps me relax after stressful days.", "score": 0.9},
    ]
    answer = synthesize_answer("How do Jon and Gina both like to destress?", contexts)
    assert "danc" in answer.lower()


def test_explicit_identity_label_beats_related_identity_sentence():
    contexts = [
        {"text": "Caroline: Painting helps me explore my identity and be true to myself.", "score": 1.0},
        {"text": "Caroline: I'm a transgender woman and coming out changed my life.", "score": 0.7},
    ]
    answer = synthesize_answer("What is Caroline's identity?", contexts)
    assert answer.lower() == "transgender woman"


def test_favorite_value_is_extracted_compactly():
    contexts = [
        {"text": "Gina: My favorite style of dance is Contemporary.", "score": 0.8},
        {"text": "Gina: Dance is a huge part of my life.", "score": 1.0},
    ]
    answer = synthesize_answer("What is Gina's favorite style of dance?", contexts)
    assert answer.lower() == "contemporary"


def test_generic_activity_list_synthesis():
    contexts = [
        {"text": "Melanie: I've been running farther to de-stress.", "score": 0.9},
        {"text": "Melanie: I signed up for a pottery class because it feels therapeutic.", "score": 0.8},
    ]
    answer = synthesize_answer("What does Melanie do to destress?", contexts)
    assert "running" in answer.lower()
    assert "pottery" in answer.lower()


def test_generic_event_participation_synthesis():
    contexts = [
        {"text": "Caroline: I attended a pride parade downtown.", "score": 0.9},
        {"text": "Caroline: I went to a support group last week.", "score": 0.8},
    ]
    answer = synthesize_answer("What events has Caroline participated in?", contexts)
    assert "pride parade" in answer.lower()
    assert "support group" in answer.lower()


def test_transgender_topic_does_not_imply_identity_question():
    contexts = [
        {"text": "Caroline is a transgender woman.", "score": 0.5},
        {"text": "Caroline: I'm going to a transgender conference in July 2023.", "score": 0.9},
    ]
    answer = synthesize_answer("When is Caroline going to the transgender conference?", contexts)
    assert "July" in answer and "2023" in answer


def test_duration_beats_session_date_for_how_long_question():
    contexts = [
        {
            "text": "---\nsession_date: 13 September 2023\n---\nCaroline: I've had this group of friends for 4 years.",
            "score": 0.9,
        }
    ]
    answer = synthesize_answer("How long has Caroline had this group of friends for?", contexts)
    assert answer.lower() == "4 years"


def test_location_list_handles_lowercase_places_and_strips_time_modifiers():
    contexts = [
        {"text": "Melanie: We camped at the beach last summer.", "score": 0.9},
        {"text": "Melanie: We camped in the forest this spring.", "score": 0.8},
    ]
    answer = synthesize_answer("Where has Melanie camped?", contexts)
    assert "beach" in answer.lower()
    assert "forest" in answer.lower()
    assert "last summer" not in answer.lower()
    assert "this spring" not in answer.lower()


def test_shared_city_cleanup_keeps_city_not_time_modifier():
    contexts = [
        {"text": "Jean: I visited Rome last spring.", "score": 0.9},
        {"text": "John: I traveled to Rome last year.", "score": 0.9},
    ]
    answer = synthesize_answer("Which city have both Jean and John visited?", contexts)
    assert answer.lower() == "rome"
