from pathlib import Path

from ultimate_memory.config import (
    BasicMemoryConfig, DashboardConfig, Neo4jConfig, PathsConfig,
    QdrantConfig, RetrievalConfig, Settings,
)
from ultimate_memory.models import AtomicMemory, MemoryType
from ultimate_memory.router import MemoryRouter
from ultimate_memory.temporal_context import temporal_contexts


def make_settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    vault.mkdir()
    return Settings(
        paths=PathsConfig(repo_root=tmp_path, basic_memory_vault=vault, private_store=tmp_path / "private"),
        retrieval=RetrievalConfig(
            collection_name="temporal-context",
            embedding_model="BAAI/bge-small-en-v1.5",
            chunk_chars=300, chunk_overlap=40, default_limit=5, bootstrap_token_budget_chars=2000,
        ),
        qdrant=QdrantConfig(url="http://127.0.0.1:1"),
        neo4j=Neo4jConfig(uri="bolt://127.0.0.1:1", user="neo4j", password="password"),
        basic_memory=BasicMemoryConfig(project="main", cli="basic-memory-missing"),
        dashboard=DashboardConfig(host="127.0.0.1", port=8787),
    )


def test_temporal_context_renderer_preserves_provenance():
    rendered = temporal_contexts([{
        "id": "x",
        "text": "Alice moved to Paris.",
        "provenance": {"event_time": "2024-03-15", "source": "atomic-memory"},
    }])
    assert rendered[0]["text"].startswith("[Memory date: 2024-03-15]")
    assert rendered[0]["provenance"]["source"] == "atomic-memory"
    assert rendered[0]["provenance"]["temporal_context_rendered"] is True


def test_temporal_answer_sees_event_date_but_factual_answer_does_not(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    router.store.upsert_atom(
        AtomicMemory(
            id="alice-move",
            text="Alice moved to Paris.",
            memory_type=MemoryType.FACT,
            entities=["Alice", "Paris"],
            event_time="2024-03-15",
            metadata={"event_time": "2024-03-15"},
        )
    )

    temporal = router.answer("When did Alice move to Paris?", limit=5)
    assert any("[Memory date: 2024-03-15]" in text for text in temporal["contexts_used"])
    assert "2024" in temporal["answer"]

    factual = router.answer("Where did Alice move?", limit=5)
    assert all("[Memory date:" not in text for text in factual["contexts_used"])
