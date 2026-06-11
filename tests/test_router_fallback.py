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
from ultimate_memory.models import ReflectionPayload
from ultimate_memory.router import MemoryRouter


def make_settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Ovindu Preferences.md").write_text(
        "# Ovindu Preferences\n\nOvindu likes direct implementation and memory that reduces repeated explanations.",
        encoding="utf-8",
    )
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


def test_router_fallback_searches_markdown(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    router.index_vault()

    result = router.search("repeated explanations")

    assert result["mode"] == "fallback"
    assert result["results"]
    assert "repeated explanations" in result["results"][0]["text"]


def test_ingest_log_stays_private(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))

    result = router.ingest_log(
        client="codex",
        session_id="abc123",
        transcript_or_path="We learned that memory should be shared.",
    )

    assert Path(result["path"]).exists()
    assert "logs" in result["path"]
    assert not any("abc123" in path.name for path in (tmp_path / "vault").glob("*.md"))


def test_private_log_can_rank_above_noisy_notes(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))
    router.index_vault()
    router.ingest_log(
        client="codex",
        session_id="rank-test",
        transcript_or_path="Codex smoke test privately ingest session logs exact recall marker.",
    )

    result = router.search("Codex smoke test privately ingest session logs", limit=3)

    assert result["results"][0]["memory_type"] == "log"
    assert "exact recall marker" in result["results"][0]["text"]


def test_reflect_writes_or_stages(tmp_path):
    router = MemoryRouter(make_settings(tmp_path))

    result = router.reflect(
        ReflectionPayload(
            summary="Ovindu wants smart shared memory.",
            preferences=["Ovindu wants fewer repeated explanations."],
            source_refs=["test"],
        )
    )

    assert "confidence" in result
    assert result["canonical_written"] is True
