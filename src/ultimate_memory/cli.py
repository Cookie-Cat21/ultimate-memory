from __future__ import annotations

import json

import typer

from .models import ReflectionPayload
from .router import MemoryRouter

app = typer.Typer(help="Ultimate Memory local maintenance CLI.")


def emit(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


@app.command()
def health() -> None:
    """Show Basic Memory/Qdrant/Neo4j health."""
    emit(MemoryRouter().health())


@app.command("index-vault")
def index_vault() -> None:
    """Index the existing Basic Memory vault into local stores."""
    emit(MemoryRouter().index_vault())


@app.command()
def search(query: str, limit: int = 8) -> None:
    """Search all available memory layers."""
    emit(MemoryRouter().search(query=query, limit=limit))


@app.command()
def bootstrap(task: str, project_path: str | None = None) -> None:
    """Print the bootstrap context packet for a task."""
    emit(MemoryRouter().bootstrap(task=task, project_path=project_path))


@app.command("ingest-log")
def ingest_log(
    client: str,
    session_id: str,
    transcript_or_path: str,
    project_path: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Store and index a session log."""
    emit(
        MemoryRouter().ingest_log(
            client=client,
            session_id=session_id,
            transcript_or_path=transcript_or_path,
            project_path=project_path,
            tags=tags,
        )
    )


@app.command()
def reflect(payload_json: str) -> None:
    """Submit a ReflectionPayload JSON document."""
    payload = ReflectionPayload(**json.loads(payload_json))
    emit(MemoryRouter().reflect(payload))


@app.command()
def atoms(
    query: str | None = None,
    limit: int = 20,
    include_superseded: bool = False,
) -> None:
    """List or search typed atomic memories."""
    emit(
        MemoryRouter().list_atoms(
            query=query,
            limit=limit,
            include_superseded=include_superseded,
        )
    )


@app.command()
def consolidate(dry_run: bool = True) -> None:
    """Merge near-duplicate atomic memories (dry-run by default)."""
    emit(MemoryRouter().consolidate(dry_run=dry_run))
