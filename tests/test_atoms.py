"""Tests for bi-temporal atomic memory, contradiction, salience, and consolidate."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ultimate_memory.atoms import (
    atoms_from_reflection,
    blend_scores,
    compute_salience,
    contradiction_score,
    content_hash,
    group_near_duplicates,
    has_supersession_cue,
    jaccard,
    tokenize,
)
from ultimate_memory.config import (
    BasicMemoryConfig,
    DashboardConfig,
    Neo4jConfig,
    PathsConfig,
    QdrantConfig,
    RetrievalConfig,
    Settings,
)
from ultimate_memory.models import AtomicMemory, MemoryType, ReflectionPayload
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


class TestAtomHeuristics:
    def test_tokenize_filters_stopwords(self):
        tokens = tokenize("We decided to use Redis for caching always")
        assert "redis" in tokens
        assert "caching" in tokens
        assert "the" not in tokens
        assert "use" not in tokens

    def test_jaccard_identical(self):
        a = tokenize("prefer postgres for primary storage")
        assert jaccard(a, a) == 1.0

    def test_contradiction_detects_replacement(self):
        old = "We decided to use Redis for caching going forward."
        new = "We decided to use Memcached instead of Redis for caching."
        assert has_supersession_cue(new)
        assert contradiction_score(new, old) >= 0.58

    def test_contradiction_low_for_unrelated(self):
        assert contradiction_score("Prefer dark mode in the IDE", "Use Postgres for OLTP") < 0.3

    def test_salience_decays_with_age(self):
        now = datetime.now(UTC)
        fresh = compute_salience(
            "fact",
            created_at=now.isoformat(),
            last_accessed=None,
            access_count=0,
            valid_until=None,
            now=now,
        )
        old = compute_salience(
            "fact",
            created_at=(now - timedelta(days=60)).isoformat(),
            last_accessed=None,
            access_count=0,
            valid_until=None,
            now=now,
        )
        assert fresh > old

    def test_salience_zero_when_invalid(self):
        score = compute_salience(
            "preference",
            created_at=datetime.now(UTC).isoformat(),
            last_accessed=None,
            access_count=5,
            valid_until=datetime.now(UTC).isoformat(),
        )
        assert score == 0.0

    def test_salience_reinforcement(self):
        now = datetime.now(UTC)
        cold = compute_salience(
            "preference",
            created_at=now.isoformat(),
            last_accessed=now.isoformat(),
            access_count=0,
            valid_until=None,
            now=now,
        )
        hot = compute_salience(
            "preference",
            created_at=now.isoformat(),
            last_accessed=now.isoformat(),
            access_count=20,
            valid_until=None,
            now=now,
        )
        assert hot > cold

    def test_blend_scores(self):
        assert 0.0 < blend_scores(1.0, 0.0) < 1.0
        assert blend_scores(1.0, 1.0) == 1.0

    def test_atoms_from_reflection(self):
        payload = ReflectionPayload(
            summary="session",
            facts=["The root cause was a bad timeout value in config."],
            decisions=["We decided to use Redis for session caching."],
            preferences=["I prefer environment variables for secrets."],
            procedures=["Steps: migrate, restart, verify logs carefully."],
            source_refs=["session:1"],
        )
        atoms = atoms_from_reflection(payload, project_path="/proj/demo", entities=["Redis"])
        assert len(atoms) == 4
        types = {a.memory_type for a in atoms}
        assert types == {
            MemoryType.FACT,
            MemoryType.DECISION,
            MemoryType.PREFERENCE,
            MemoryType.PROCEDURE,
        }
        assert all(a.content_hash for a in atoms)
        assert content_hash(atoms[0].text) == atoms[0].content_hash


class TestAtomLifecycle:
    def test_reflect_creates_atoms(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        result = router.reflect(
            ReflectionPayload(
                summary="Caching decision",
                decisions=["We decided to use Redis for caching going forward."],
                preferences=["I prefer we always use environment variables for secrets."],
                source_refs=["test"],
            ),
            project_path=str(tmp_path / "demo-project"),
        )
        assert result["atoms"]["count"] >= 2
        atoms = router.list_atoms(limit=20)
        assert atoms["count"] >= 2
        types = {a["memory_type"] for a in atoms["atoms"]}
        assert "decision" in types
        assert "preference" in types

    def test_contradiction_auto_supersedes(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.reflect(
            ReflectionPayload(
                summary="v1",
                decisions=["We decided to use Redis for caching going forward."],
                source_refs=["s1"],
            )
        )
        result = router.reflect(
            ReflectionPayload(
                summary="v2",
                decisions=["We decided to use Memcached instead of Redis for caching."],
                source_refs=["s2"],
            )
        )
        assert result["atoms"]["superseded"]
        active = router.list_atoms(memory_types=["decision"], limit=20)
        texts = " ".join(a["text"] for a in active["atoms"])
        assert "Memcached" in texts
        # Old Redis decision should be inactive
        superseded = router.list_atoms(
            query="Redis",
            memory_types=["decision"],
            include_superseded=True,
            limit=20,
        )
        old = [a for a in superseded["atoms"] if "Redis" in a["text"] and "Memcached" not in a["text"]]
        assert old
        assert old[0]["valid_until"] is not None or old[0]["superseded_by"] is not None

    def test_duplicate_atom_reinforces(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        payload = ReflectionPayload(
            summary="dup",
            preferences=["I prefer environment variables for secrets always."],
            source_refs=["a"],
        )
        router.reflect(payload)
        second = router.reflect(payload)
        assert second["atoms"]["duplicates"]
        assert second["atoms"]["count"] == 0
        atoms = router.list_atoms(memory_types=["preference"], limit=10)
        assert atoms["count"] == 1
        assert atoms["atoms"][0]["access_count"] >= 1

    def test_search_returns_atoms_and_touches(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.reflect(
            ReflectionPayload(
                summary="searchable",
                facts=["The root cause was a misconfigured timeout in settings."],
                source_refs=["t"],
            )
        )
        before = router.list_atoms(query="timeout", limit=5)["atoms"][0]["access_count"]
        result = router.search("misconfigured timeout", limit=5)
        assert result["atoms_considered"] >= 1
        assert any(r["id"].startswith("atom:") for r in result["results"]) or any(
            r["provenance"].get("source") == "atomic-memory" for r in result["results"]
        )
        after = router.list_atoms(query="timeout", limit=5)["atoms"][0]["access_count"]
        assert after >= before

    def test_bootstrap_prefers_preferences(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.reflect(
            ReflectionPayload(
                summary="boot",
                preferences=["I prefer concise answers and direct implementation."],
                procedures=["Steps: run tests then commit then push carefully."],
                decisions=["We decided to keep memory local-first always."],
                source_refs=["b"],
            )
        )
        packet = router.bootstrap("implement memory feature", project_path=str(tmp_path / "um"))
        assert packet["composition"]["preferences"] >= 1
        assert packet["context_packet"]

    def test_consolidate_merges_near_duplicates(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        router.store.upsert_atom(
            AtomicMemory(
                id="atom-a",
                text="Prefer environment variables for secrets in all services.",
                memory_type=MemoryType.PREFERENCE,
            )
        )
        router.store.upsert_atom(
            AtomicMemory(
                id="atom-b",
                text="Prefer environment variables for secrets in all services today.",
                memory_type=MemoryType.PREFERENCE,
            )
        )
        dry = router.consolidate(dry_run=True)
        assert dry["groups_found"] >= 1
        live = router.consolidate(dry_run=False)
        assert live["atoms_merged"] >= 1
        active = router.list_atoms(memory_types=["preference"], limit=10)
        assert active["count"] == 1

    def test_group_near_duplicates_helper(self):
        atoms = [
            AtomicMemory(
                id="1",
                text="Use docker compose for local neo4j and qdrant.",
                memory_type=MemoryType.PROCEDURE,
            ),
            AtomicMemory(
                id="2",
                text="Use docker compose for local neo4j and qdrant services.",
                memory_type=MemoryType.PROCEDURE,
            ),
            AtomicMemory(
                id="3",
                text="Prefer dark theme in the editor.",
                memory_type=MemoryType.PREFERENCE,
            ),
        ]
        groups = group_near_duplicates(atoms, threshold=0.5)
        assert len(groups) == 1
        assert {a.id for a in groups[0]} == {"1", "2"}
