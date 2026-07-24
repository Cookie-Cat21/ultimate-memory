"""Tests for multi-hop memory retrieval."""

from __future__ import annotations

from pathlib import Path

from ultimate_memory.config import (
    BasicMemoryConfig,
    DashboardConfig,
    Neo4jConfig,
    PathsConfig,
    QdrantConfig,
    RetrievalConfig,
    Settings,
)
from ultimate_memory.hops import (
    MAX_HOP_SEARCHES,
    build_hop_queries,
    extract_capitalized_entities,
    extract_hop_entities,
    merge_contexts,
)
from ultimate_memory.models import AtomicMemory, MemoryType
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


class TestHopHelpers:
    def test_extract_capitalized_entities_skips_question_words(self):
        ents = extract_capitalized_entities("What does Elena's sister do for work?")
        assert "Elena" in ents
        assert "What" not in ents

    def test_extract_hop_entities_from_results_and_atoms(self):
        results = [
            {
                "text": "Elena's sister is named Fiona.",
                "title": "fact: Elena's sister is named Fiona.",
                "provenance": {"entities": ["Elena", "Fiona"]},
            }
        ]
        entities = extract_hop_entities(
            "What does Elena's sister do for work?",
            results,
        )
        assert "Fiona" in entities
        assert "Elena" in entities

    def test_build_hop_queries_prefers_bridge_entity(self):
        queries = build_hop_queries(
            "What does Elena's sister do for work?",
            ["Elena", "Fiona"],
            max_queries=2,
        )
        assert queries
        assert queries[0].startswith("Fiona")
        assert "work" in queries[0] or "job" in queries[0]

    def test_merge_contexts_dedupes(self):
        a = {"text": "Elena's sister is named Fiona."}
        b = {"text": "Fiona works as a pilot for Delta Airlines."}
        merged = merge_contexts([a], [a, b])
        assert len(merged) == 2
        assert merged[1]["text"] == b["text"]

    def test_max_hop_searches_cap(self):
        assert MAX_HOP_SEARCHES == 3


class TestMultiHopAnswer:
    def test_elena_fiona_job_question(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.store.upsert_atom(
            AtomicMemory(
                id="elena-sister",
                text="Elena's sister is named Fiona.",
                memory_type=MemoryType.FACT,
                entities=["Elena", "Fiona"],
            )
        )
        router.store.upsert_atom(
            AtomicMemory(
                id="fiona-job",
                text="Fiona works as a pilot for Delta Airlines.",
                memory_type=MemoryType.FACT,
                entities=["Fiona", "Delta Airlines"],
            )
        )

        result = router.answer("What does Elena's sister do for work?", limit=5)

        assert result["hop_searches"], "expected at least one follow-up hop search"
        assert "Fiona" in result["hop_entities"]
        assert any("pilot" in text.lower() for text in result["contexts_used"])
        assert "pilot" in result["answer"].lower()

    def test_single_hop_skips_when_no_bridge_entities(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.store.upsert_atom(
            AtomicMemory(
                id="alice-job",
                text="Alice works as a nurse in Seattle.",
                memory_type=MemoryType.FACT,
                entities=["Alice"],
            )
        )

        result = router.answer("What does Alice do for work?", limit=5)

        assert result["answer"]
        assert "nurse" in result["answer"].lower()
        assert len(result["hop_searches"]) <= MAX_HOP_SEARCHES
