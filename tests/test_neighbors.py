from pathlib import Path

from ultimate_memory.config import (
    BasicMemoryConfig, DashboardConfig, Neo4jConfig, PathsConfig,
    QdrantConfig, RetrievalConfig, Settings,
)
from ultimate_memory.models import AtomicMemory, MemoryType
from ultimate_memory.router import MemoryRouter


def make_settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    vault.mkdir()
    return Settings(
        paths=PathsConfig(repo_root=tmp_path, basic_memory_vault=vault, private_store=tmp_path / "private"),
        retrieval=RetrievalConfig(
            collection_name="neighbors",
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


def turn(atom_id: str, dia_id: str, text: str, *, session: str = "s1", project: str = "/p") -> AtomicMemory:
    return AtomicMemory(
        id=atom_id,
        text=text,
        memory_type=MemoryType.FACT,
        project_path=project,
        entities=["Alice"],
        metadata={"session_id": session, "dia_id": dia_id, "speaker": "Alice"},
    )


def test_dialogue_neighbors_are_local_and_ordered(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    for atom in [
        turn("t1", "D1:1", "Alice introduced the project."),
        turn("t2", "D1:2", "Alice said the launch was Tuesday."),
        turn("t3", "D1:3", "Alice discussed the venue."),
        turn("other-session", "D1:1", "Unrelated session.", session="s2"),
        turn("other-project", "D1:3", "Unrelated project.", project="/other"),
    ]:
        router.store.upsert_atom(atom)

    neighbors = router.store.dialogue_neighbors("s1", "D1:2", radius=1, project_path="/p")
    assert [item.id for item in neighbors] == ["t1", "t3"]


def test_answer_reports_neighbor_expansion(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    router.store.upsert_atom(turn("t1", "D1:1", "Alice said the release date is Tuesday."))
    router.store.upsert_atom(turn("t2", "D1:2", "Alice discussed Project Zephyr readiness."))
    router.store.upsert_atom(turn("t3", "D1:3", "Alice said testing is complete."))

    result = router.answer("What did Alice discuss about Project Zephyr?", project_path="/p", limit=3)
    assert result["neighbor_contexts"] >= 1
    assert any("Tuesday" in text or "testing" in text for text in result["contexts_used"])
