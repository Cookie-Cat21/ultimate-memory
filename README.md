# Ultimate Memory

Local-first MCP memory router for Claude Code and Codex.

The router keeps `C:\Users\Ovindu\basic-memory` as the canonical Obsidian-readable vault, then adds:

- Qdrant vector retrieval over vault notes and private full logs.
- Neo4j graph memory for entities, relations, provenance, and supersession.
- MCP tools that Claude and Codex can both call.
- A private audit/log store under `.ultimate-memory`.
- A tiny dashboard for health, staged candidates, audit events, and retrieval smoke tests.

## Quick Start

```powershell
cd "C:\Users\Ovindu\Documents\Pet Projects\ultimate-memory"
docker compose up -d
uv sync --extra dev
uv run ultimate-memory health
uv run ultimate-memory index-vault
uv run ultimate-memory-router
```

If Docker Desktop is not running, the router still starts and falls back to Basic Memory plus direct markdown search.

## MCP Tools

- `memory_bootstrap(task, project_path?)`
- `memory_search(query, project_path?, memory_types?, limit?)`
- `memory_ingest_log(client, session_id, project_path?, transcript_or_path, tags?)`
- `memory_reflect(source_refs, summary, facts, preferences, decisions, procedures, open_questions?)`
- `memory_graph_query(entity_or_topic, depth?)`
- `memory_promote(candidate_id)`
- `memory_supersede(old_ref, new_fact, reason, source_refs)`

## Client Wiring

The intended MCP command is:

```powershell
uv run --project "C:\Users\Ovindu\Documents\Pet Projects\ultimate-memory" ultimate-memory-router
```

Codex and Claude should call `memory_bootstrap` at session start, `memory_search` as needed, and `memory_reflect` at session end.

Claude also has a `SessionEnd` hook wired to `scripts/ingest-claude-session.ps1`, which ingests the full transcript path into the private local log store.

## Docker Router Mode

Claude and Codex use stdio on the Windows host. Docker mode is available for service-style clients:

```powershell
docker compose --profile router up -d
```

The streamable HTTP MCP endpoint is `http://127.0.0.1:8788/mcp`.
