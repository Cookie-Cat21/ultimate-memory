"""Tests for temporal as_of retrieval and past-tense query handling."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ultimate_memory.atoms import atom_valid_at, is_temporal_query
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
from ultimate_memory.store import LocalStore


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


class TestTemporalHelpers:
    def test_is_temporal_query_detects_past_cues(self):
        assert is_temporal_query("where did Alex live before moving")
        assert is_temporal_query("what city did they used to work in")
        assert is_temporal_query("previously we stored sessions on disk")
        assert not is_temporal_query("current deployment region")

    def test_atom_valid_at(self):
        t0 = "2024-01-01T00:00:00+00:00"
        t1 = "2024-06-01T00:00:00+00:00"
        t2 = "2024-12-01T00:00:00+00:00"
        atom = AtomicMemory(
            id="a1",
            text="Lived in Austin",
            memory_type=MemoryType.FACT,
            valid_from=t0,
            valid_until=t2,
        )
        assert atom_valid_at(atom, t1)
        assert not atom_valid_at(atom, "2023-06-01T00:00:00+00:00")
        assert not atom_valid_at(atom, t2)


class TestAsOfFiltering:
    def test_list_active_atoms_as_of(self, tmp_path):
        settings = make_settings(tmp_path)
        store = LocalStore(
            settings.sqlite_dir,
            settings.audit_dir,
            settings.candidates_dir,
        )
        old_from = "2023-01-01T00:00:00+00:00"
        old_until = "2024-06-01T00:00:00+00:00"
        new_from = "2024-06-01T00:00:00+00:00"

        store.upsert_atom(
            AtomicMemory(
                id="atom-old",
                text="Alex lived in Portland before relocating.",
                memory_type=MemoryType.FACT,
                valid_from=old_from,
                valid_until=old_until,
                superseded_by="atom-new",
            )
        )
        store.upsert_atom(
            AtomicMemory(
                id="atom-new",
                text="Alex lives in Seattle after relocating.",
                memory_type=MemoryType.FACT,
                valid_from=new_from,
            )
        )

        current = store.list_active_atoms(limit=10)
        assert len(current) == 1
        assert "Seattle" in current[0].text

        mid_2023 = store.list_active_atoms(as_of="2023-06-01T00:00:00+00:00", limit=10)
        assert len(mid_2023) == 1
        assert "Portland" in mid_2023[0].text

        mid_2024 = store.list_active_atoms(as_of="2024-08-01T00:00:00+00:00", limit=10)
        assert len(mid_2024) == 1
        assert "Seattle" in mid_2024[0].text

    def test_search_atoms_as_of(self, tmp_path):
        settings = make_settings(tmp_path)
        store = LocalStore(
            settings.sqlite_dir,
            settings.audit_dir,
            settings.candidates_dir,
        )
        store.upsert_atom(
            AtomicMemory(
                id="atom-old",
                text="Office was in Boston before the move.",
                memory_type=MemoryType.FACT,
                valid_from="2022-01-01T00:00:00+00:00",
                valid_until="2024-01-01T00:00:00+00:00",
            )
        )
        store.upsert_atom(
            AtomicMemory(
                id="atom-new",
                text="Office is in Chicago after the move.",
                memory_type=MemoryType.FACT,
                valid_from="2024-01-01T00:00:00+00:00",
            )
        )

        past = store.search_atoms("Boston office", as_of="2023-06-01T00:00:00+00:00", limit=5)
        assert any("Boston" in a.text for a in past)
        assert not any("Chicago" in a.text for a in past)

        present = store.search_atoms("Chicago office", limit=5)
        assert any("Chicago" in a.text for a in present)


class TestTemporalSearchRouter:
    def test_used_to_includes_superseded(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        t_old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        t_new = datetime.now(UTC).isoformat()

        router.store.upsert_atom(
            AtomicMemory(
                id="alex-portland",
                text="Alex lived in Portland before relocating to the west coast.",
                memory_type=MemoryType.FACT,
                valid_from=t_old,
                valid_until=t_new,
                superseded_by="alex-seattle",
            )
        )
        router.store.upsert_atom(
            AtomicMemory(
                id="alex-seattle",
                text="Alex lives in Seattle instead of Portland after relocating.",
                memory_type=MemoryType.FACT,
                valid_from=t_new,
            )
        )

        result = router.search("where did Alex used to live in Portland", limit=8)
        assert result["temporal_query"] is True
        assert result["atoms_considered"] >= 1
        superseded = router.list_atoms(
            query="Portland",
            include_superseded=True,
            limit=5,
        )
        assert any("Portland" in a["text"] for a in superseded["atoms"])

    def test_as_of_search_point_in_time(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        t_old = (datetime.now(UTC) - timedelta(days=400)).isoformat()
        t_mid = (datetime.now(UTC) - timedelta(days=200)).isoformat()
        t_new = datetime.now(UTC).isoformat()

        router.store.upsert_atom(
            AtomicMemory(
                id="loc-old",
                text="Team headquarters was in Denver Colorado.",
                memory_type=MemoryType.FACT,
                valid_from=t_old,
                valid_until=t_mid,
            )
        )
        router.store.upsert_atom(
            AtomicMemory(
                id="loc-new",
                text="Team headquarters is in Austin Texas.",
                memory_type=MemoryType.FACT,
                valid_from=t_mid,
            )
        )

        as_of = (datetime.now(UTC) - timedelta(days=300)).isoformat()
        atoms = router.list_atoms(query="Denver headquarters", as_of=as_of, limit=5)
        assert atoms["count"] >= 1
        assert any("Denver" in a["text"] for a in atoms["atoms"])

        current = router.list_atoms(query="headquarters", limit=5)
        assert any("Austin" in a["text"] for a in current["atoms"])


class TestIngestEventTime:
    def test_ingest_log_stamps_event_time_from_header(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        transcript = "\n".join(
            [
                "---",
                "created_at: 2024-03-15T10:00:00+00:00",
                "---",
                "",
                "User: Important: the fix was to increase the connection timeout.",
            ]
        )
        result = router.ingest_log(
            client="test",
            session_id="sess-temporal",
            transcript_or_path=transcript,
        )
        assert result["reflection"] is not None
        atoms = router.list_atoms(query="connection timeout", limit=5)
        if atoms["count"]:
            atom = atoms["atoms"][0]
            assert atom.get("event_time") == "2024-03-15T10:00:00+00:00"
            assert atom["valid_from"].startswith("2024-03-15")


class TestDialogueTurnIngest:
    def test_ingest_log_indexes_dialogue_turns(self, tmp_path):
        router = MemoryRouter(make_settings(tmp_path))
        transcript = "\n".join(
            [
                "---",
                "created_at: 2024-03-15T10:00:00+00:00",
                "---",
                "",
                "[D1:1] Caroline: Hi Melanie!",
                "[D1:2] Melanie: I painted a bowl for my friend's birthday last year.",
                "[D1:3] Caroline: That necklace is from my grandmother in Sweden and means a lot to me.",
            ]
        )
        result = router.ingest_log(
            client="test",
            session_id="sess-dia",
            transcript_or_path=transcript,
        )
        assert result["turn_chunks"] == 3
        assert result["turn_atoms"] == 2

        search = router.search("necklace grandmother Sweden", limit=8)
        texts = " ".join(item["text"] for item in search["results"])
        assert "Sweden" in texts
        assert "D1:3" in texts

        atoms = router.list_atoms(query="painted bowl", limit=5)
        assert atoms["count"] >= 1
        assert any("Melanie" in atom["text"] for atom in atoms["atoms"])
