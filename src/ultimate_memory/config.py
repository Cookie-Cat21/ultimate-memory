from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


# Falls back to <repo_root>/config/ultimate-memory.toml (copy
# config/ultimate-memory.toml.example there and edit the paths) unless
# ULTIMATE_MEMORY_CONFIG points somewhere else.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "ultimate-memory.toml"


@dataclass(frozen=True)
class PathsConfig:
    repo_root: Path
    basic_memory_vault: Path
    private_store: Path


@dataclass(frozen=True)
class RetrievalConfig:
    collection_name: str
    embedding_model: str
    chunk_chars: int
    chunk_overlap: int
    default_limit: int
    bootstrap_token_budget_chars: int


@dataclass(frozen=True)
class QdrantConfig:
    url: str


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class BasicMemoryConfig:
    project: str
    cli: str


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    retrieval: RetrievalConfig
    qdrant: QdrantConfig
    neo4j: Neo4jConfig
    basic_memory: BasicMemoryConfig
    dashboard: DashboardConfig

    @property
    def logs_dir(self) -> Path:
        return self.paths.private_store / "logs"

    @property
    def audit_dir(self) -> Path:
        return self.paths.private_store / "audit"

    @property
    def candidates_dir(self) -> Path:
        return self.paths.private_store / "candidates"

    @property
    def sqlite_dir(self) -> Path:
        return self.paths.private_store / "sqlite"


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or Path(os.environ.get("ULTIMATE_MEMORY_CONFIG", DEFAULT_CONFIG))
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    paths = data["paths"]
    retrieval = data["retrieval"]
    qdrant = data["qdrant"]
    neo4j = data["neo4j"]
    basic_memory = data["basic_memory"]
    dashboard = data["dashboard"]

    settings = Settings(
        paths=PathsConfig(
            repo_root=Path(paths["repo_root"]),
            basic_memory_vault=Path(paths["basic_memory_vault"]),
            private_store=Path(paths["private_store"]),
        ),
        retrieval=RetrievalConfig(
            collection_name=retrieval["collection_name"],
            embedding_model=retrieval["embedding_model"],
            chunk_chars=int(retrieval["chunk_chars"]),
            chunk_overlap=int(retrieval["chunk_overlap"]),
            default_limit=int(retrieval["default_limit"]),
            bootstrap_token_budget_chars=int(retrieval["bootstrap_token_budget_chars"]),
        ),
        qdrant=QdrantConfig(url=qdrant["url"]),
        neo4j=Neo4jConfig(
            uri=neo4j["uri"],
            user=neo4j["user"],
            password=neo4j["password"],
        ),
        basic_memory=BasicMemoryConfig(
            project=basic_memory.get("project", "main"),
            cli=basic_memory.get("cli", "basic-memory"),
        ),
        dashboard=DashboardConfig(
            host=dashboard.get("host", "127.0.0.1"),
            port=int(dashboard.get("port", 8787)),
        ),
    )
    ensure_private_dirs(settings)
    return settings


def ensure_private_dirs(settings: Settings) -> None:
    for folder in (
        settings.paths.private_store,
        settings.logs_dir,
        settings.audit_dir,
        settings.candidates_dir,
        settings.sqlite_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

