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
from ultimate_memory.models import AtomicMemory, MemoryType
from ultimate_memory.router import MemoryRouter


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


def test_entity_channel_recovers_low_lexical_overlap(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    router.store.upsert_atom(
        AtomicMemory(
            id="elena-role",
            text="Elena is a pilot in Seattle.",
            memory_type=MemoryType.FACT,
            entities=["Elena", "Seattle"],
            project_path="/projects/a",
        )
    )
    router.store.upsert_atom(
        AtomicMemory(
            id="fiona-role",
            text="Fiona is a designer in Boston.",
            memory_type=MemoryType.FACT,
            entities=["Fiona", "Boston"],
            project_path="/projects/a",
        )
    )

    result = router.search(
        "What occupation does Elena have?",
        project_path="/projects/a",
        limit=5,
    )

    elena = next(item for item in result["results"] if item["id"] == "elena-role")
    assert elena["provenance"]["entity_match"] is True
    assert result["entity_atoms_considered"] >= 1


def test_entity_channel_respects_project_scope(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    router.store.upsert_atom(
        AtomicMemory(
            id="a",
            text="Elena uses PostgreSQL.",
            memory_type=MemoryType.FACT,
            entities=["Elena"],
            project_path="/projects/a",
        )
    )
    router.store.upsert_atom(
        AtomicMemory(
            id="b",
            text="Elena uses MySQL.",
            memory_type=MemoryType.FACT,
            entities=["Elena"],
            project_path="/projects/b",
        )
    )

    result = router.search("What database does Elena use?", project_path="/projects/a", limit=10)
    ids = {item["id"] for item in result["results"]}
    assert "a" in ids
    assert "b" not in ids
