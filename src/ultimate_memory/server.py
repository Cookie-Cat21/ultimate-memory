from __future__ import annotations

import argparse
import os
from typing import Annotated

from mcp.server.fastmcp import FastMCP

from .models import ReflectionPayload
from .router import MemoryRouter

mcp = FastMCP(
    "ultimate-memory",
    host=os.environ.get("ULTIMATE_MEMORY_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("ULTIMATE_MEMORY_MCP_PORT", "8788")),
    streamable_http_path="/mcp",
)
router = MemoryRouter()


@mcp.tool()
def memory_bootstrap(task: str, project_path: str | None = None) -> dict:
    """Return a compact context packet for a task before asking the user to re-explain."""
    return router.bootstrap(task=task, project_path=project_path)


@mcp.tool()
def memory_search(
    query: str,
    project_path: str | None = None,
    memory_types: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Search Basic Memory, local full logs, vector memory, and graph memory."""
    return router.search(query=query, project_path=project_path, memory_types=memory_types, limit=limit)


@mcp.tool()
def memory_ingest_log(
    client: str,
    session_id: str,
    transcript_or_path: str,
    project_path: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Store and index a full local session log without writing it into Obsidian."""
    return router.ingest_log(
        client=client,
        session_id=session_id,
        project_path=project_path,
        transcript_or_path=transcript_or_path,
        tags=tags,
    )


@mcp.tool()
def memory_reflect(
    summary: str,
    source_refs: list[str] | None = None,
    facts: list[str] | None = None,
    preferences: list[str] | None = None,
    decisions: list[str] | None = None,
    procedures: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> dict:
    """Submit distilled learnings for durable memory, dedupe, audit, and graph linking."""
    payload = ReflectionPayload(
        source_refs=source_refs or [],
        summary=summary,
        facts=facts or [],
        preferences=preferences or [],
        decisions=decisions or [],
        procedures=procedures or [],
        open_questions=open_questions or [],
    )
    return router.reflect(payload)


@mcp.tool()
def memory_graph_query(entity_or_topic: str, depth: Annotated[int, "1 to 3"] = 1) -> dict:
    """Return graph neighbors, related sources, and supersession relations for a topic."""
    return router.graph_query(entity_or_topic=entity_or_topic, depth=depth)


@mcp.tool()
def memory_promote(candidate_id: str) -> dict:
    """Promote a staged low-confidence memory candidate into canonical Basic Memory."""
    return router.promote(candidate_id=candidate_id)


@mcp.tool()
def memory_supersede(
    old_ref: str,
    new_fact: str,
    reason: str,
    source_refs: list[str] | None = None,
) -> dict:
    """Record that a newer fact supersedes an older memory reference."""
    return router.supersede(
        old_ref=old_ref,
        new_fact=new_fact,
        reason=reason,
        source_refs=source_refs or [],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultimate Memory MCP router")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.environ.get("ULTIMATE_MEMORY_TRANSPORT", "stdio"),
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
