"""Focused tests for knowledge-update supersession (Carol/Gabe/preference)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultimate_memory.atoms import contradiction_score, role_conflict_boost
from ultimate_memory.config import (
    BasicMemoryConfig,
    DashboardConfig,
    Neo4jConfig,
    PathsConfig,
    QdrantConfig,
    RetrievalConfig,
    Settings,
)
from ultimate_memory.models import ReflectionPayload
from ultimate_memory.router import MemoryRouter

CAROL_OLD = "Carol lives in New York City."
CAROL_NEW = "Carol moved to Austin instead of staying in New York City."
GABE_OLD = "Gabe works as a teacher at Lincoln High."
GABE_NEW = "Gabe works as a software engineer instead of working as a teacher."
PREF_OLD = "I always prefer dark mode in the editor."
PREF_NEW = "I no longer prefer dark mode; I prefer light mode in the editor instead."


def make_settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    vault.mkdir()
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


class TestKnowledgeUpdateScores:
    def test_carol_location_contradiction_scores(self):
        boost = role_conflict_boost(CAROL_NEW, CAROL_OLD)
        score = contradiction_score(CAROL_NEW, CAROL_OLD)
        assert boost >= 0.35, f"role_boost={boost}"
        assert score >= 0.58, f"contradiction={score}"

    def test_gabe_job_contradiction_scores(self):
        boost = role_conflict_boost(GABE_NEW, GABE_OLD)
        score = contradiction_score(GABE_NEW, GABE_OLD)
        assert boost >= 0.35, f"role_boost={boost}"
        assert score >= 0.58, f"contradiction={score}"

    def test_preference_dark_to_light_scores(self):
        score = contradiction_score(PREF_NEW, PREF_OLD)
        assert score >= 0.58, f"contradiction={score}"


class TestKnowledgeUpdateSupersession:
    @pytest.fixture()
    def router(self, tmp_path: Path) -> MemoryRouter:
        return MemoryRouter(make_settings(tmp_path))

    def test_carol_supersedes_and_search_prefers_austin(self, router: MemoryRouter):
        router.reflect(
            ReflectionPayload(summary="v1", facts=[CAROL_OLD], source_refs=["w3a"])
        )
        result = router.reflect(
            ReflectionPayload(summary="v2", facts=[CAROL_NEW], source_refs=["w3b"])
        )
        assert result["atoms"]["superseded"]
        assert any("New York" in item["old_text"] for item in result["atoms"]["superseded"])

        active = router.list_atoms(memory_types=["fact"], limit=10)
        assert active["count"] == 1
        assert "Austin" in active["atoms"][0]["text"]

        search = router.search("Where does Carol live now?", limit=8)
        atom_texts = [
            r["text"]
            for r in search["results"]
            if r.get("provenance", {}).get("source") == "atomic-memory"
        ]
        assert atom_texts
        assert all("New York" not in t or "instead" in t for t in atom_texts)
        assert any("Austin" in t for t in atom_texts)

    def test_gabe_supersedes_teacher_with_engineer(self, router: MemoryRouter):
        router.reflect(
            ReflectionPayload(summary="j1", facts=[GABE_OLD], source_refs=["w6a"])
        )
        result = router.reflect(
            ReflectionPayload(summary="j2", facts=[GABE_NEW], source_refs=["w6b"])
        )
        assert result["atoms"]["superseded"]

        active = router.list_atoms(memory_types=["fact"], limit=10)
        assert active["count"] == 1
        assert "engineer" in active["atoms"][0]["text"].lower()

        superseded = router.list_atoms(
            query="teacher",
            memory_types=["fact"],
            include_superseded=True,
            limit=10,
        )
        old = [a for a in superseded["atoms"] if "teacher" in a["text"] and "engineer" not in a["text"]]
        assert old
        assert old[0]["valid_until"] is not None or old[0]["superseded_by"] is not None

    def test_preference_update_leaves_one_active_preference(self, router: MemoryRouter):
        router.reflect(
            ReflectionPayload(summary="p1", preferences=[PREF_OLD], source_refs=["p1"])
        )
        result = router.reflect(
            ReflectionPayload(summary="p2", preferences=[PREF_NEW], source_refs=["p2"])
        )
        assert result["atoms"]["superseded"]

        active = router.list_atoms(memory_types=["preference"], limit=10)
        assert active["count"] == 1
        assert "light" in active["atoms"][0]["text"].lower()

    def test_candidate_finding_with_many_distractors(self, router: MemoryRouter, tmp_path: Path):
        """Subject-hint lookup must still find Carol's prior fact among noise."""
        for i in range(15):
            router.reflect(
                ReflectionPayload(
                    summary=f"noise{i}",
                    facts=[f"Random person{i} lives in city number {i}."],
                    source_refs=[f"n{i}"],
                )
            )
        router.reflect(
            ReflectionPayload(summary="v1", facts=[CAROL_OLD], source_refs=["w3a"])
        )
        result = router.reflect(
            ReflectionPayload(summary="v2", facts=[CAROL_NEW], source_refs=["w3b"])
        )
        assert result["atoms"]["superseded"]
        active = router.list_atoms(query="Carol", memory_types=["fact"], limit=10)
        assert active["count"] == 1
        assert "Austin" in active["atoms"][0]["text"]
