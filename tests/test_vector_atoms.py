"""Unit tests for atomic memory vector embedding (no live Qdrant)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ultimate_memory.adapters.vector import VectorAdapter
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


def _retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        collection_name="test",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_chars=300,
        chunk_overlap=40,
        default_limit=5,
        bootstrap_token_budget_chars=2000,
    )


class TestVectorAtomPayload:
    def test_atom_payload_includes_kind_and_source_path(self):
        atom = AtomicMemory(
            id="atom:pref:test",
            text="Prefer concise answers.",
            memory_type=MemoryType.PREFERENCE,
            project_path="/proj/demo",
            salience=0.72,
        )
        payload = VectorAdapter._atom_payload(atom)

        assert payload["kind"] == "atom"
        assert payload["source_path"] == "atom://atom:pref:test"
        assert payload["memory_type"] == "preference"
        assert payload["salience"] == 0.72
        assert payload["project_path"] == "/proj/demo"
        assert payload["valid_until"] is None

    def test_search_result_from_atom_payload_matches_atomic_memory_shape(self):
        payload = VectorAdapter._atom_payload(
            AtomicMemory(
                id="atom:fact:abc",
                text="The timeout was misconfigured.",
                memory_type=MemoryType.FACT,
                salience=0.6,
            )
        )
        result = VectorAdapter._search_result_from_payload(payload, score=0.88)

        assert result.id == "atom:fact:abc"
        assert result.source_path == "atom://atom:fact:abc"
        assert result.provenance["source"] == "atomic-memory"
        assert result.provenance["vector"] is True
        assert result.provenance["salience"] == 0.6

    def test_search_result_skips_superseded_atoms_by_default(self):
        adapter = VectorAdapter(
            QdrantConfig(url="http://127.0.0.1:6333"),
            _retrieval_config(),
        )
        active = VectorAdapter._atom_payload(
            AtomicMemory(
                id="atom:active",
                text="Active fact.",
                memory_type=MemoryType.FACT,
            )
        )
        superseded = VectorAdapter._atom_payload(
            AtomicMemory(
                id="atom:old",
                text="Old fact.",
                memory_type=MemoryType.FACT,
                valid_until="2026-01-01T00:00:00+00:00",
            )
        )

        mock_point = MagicMock()
        mock_point.payload = active
        mock_point.score = 0.9
        mock_point_old = MagicMock()
        mock_point_old.payload = superseded
        mock_point_old.score = 0.95

        mock_response = MagicMock()
        mock_response.points = [mock_point, mock_point_old]

        with patch.object(adapter, "ensure_collection", return_value=True), patch.object(
            adapter, "embed", return_value=[[0.1, 0.2]]
        ), patch.object(adapter, "client") as mock_client:
            mock_client.return_value.query_points.return_value = mock_response
            results = adapter.search("fact", limit=5)

        assert len(results) == 1
        assert results[0].id == "atom:active"

    def test_search_result_includes_superseded_when_requested(self):
        adapter = VectorAdapter(
            QdrantConfig(url="http://127.0.0.1:6333"),
            _retrieval_config(),
        )
        superseded = VectorAdapter._atom_payload(
            AtomicMemory(
                id="atom:old",
                text="Old fact.",
                memory_type=MemoryType.FACT,
                valid_until="2026-01-01T00:00:00+00:00",
            )
        )

        mock_point = MagicMock()
        mock_point.payload = superseded
        mock_point.score = 0.95
        mock_response = MagicMock()
        mock_response.points = [mock_point]

        with patch.object(adapter, "ensure_collection", return_value=True), patch.object(
            adapter, "embed", return_value=[[0.1, 0.2]]
        ), patch.object(adapter, "client") as mock_client:
            mock_client.return_value.query_points.return_value = mock_response
            results = adapter.search("fact", limit=5, include_superseded=True)

        assert len(results) == 1
        assert results[0].id == "atom:old"


class TestRouterAtomVectorWiring:
    def test_ingest_atom_upserts_to_vector_when_ready(self, tmp_path: Path):
        router = MemoryRouter(make_settings(tmp_path))
        router._vector_ready_cache = True
        router.vector.upsert_atoms = MagicMock(return_value=1)

        result = router.reflect(
            ReflectionPayload(
                summary="vector sync",
                facts=["The root cause was a misconfigured timeout in settings."],
                source_refs=["v"],
            )
        )

        assert result["atoms"]["count"] >= 1
        assert router.vector.upsert_atoms.called
        upserted = router.vector.upsert_atoms.call_args[0][0]
        assert all(isinstance(atom, AtomicMemory) for atom in upserted)
        assert all(atom.id.startswith("atom:") for atom in upserted)

    def test_vector_unavailable_does_not_break_reflect(self, tmp_path: Path):
        router = MemoryRouter(make_settings(tmp_path))
        router._vector_ready_cache = False

        result = router.reflect(
            ReflectionPayload(
                summary="fallback",
                preferences=["I prefer environment variables for secrets."],
                source_refs=["f"],
            )
        )

        assert result["atoms"]["count"] >= 1
