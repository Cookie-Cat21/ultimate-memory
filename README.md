# Ultimate Memory

Local-first **atomic, bi-temporal** MCP memory for Claude Code and Codex.

The router keeps your Basic Memory / Obsidian vault as the human-readable canon, then adds the machinery most agent memory systems skip:

- **Atomic memories** — typed facts, decisions, preferences, and procedures (not only blob notes)
- **Bi-temporal validity** — `valid_from` / `valid_until` + `SUPERSEDES` so stale truth dies on contact
- **Contradiction-aware writes** — new atoms auto-invalidate conflicting older atoms
- **Salience ranking** — type importance × recency decay × reinforcement on access
- **Hybrid RRF retrieval** — Qdrant + SQLite FTS + vault markdown + atomic memory, fused then salience-reranked
- **Neo4j graph** — entities, atom `ABOUT` links, provenance, supersession
- **Private session logs** under `.ultimate-memory` (never dumped raw into Obsidian)
- **Confidence gating** — low-signal reflections stage as candidates until promoted
- **Consolidation** — near-duplicate atom merge with dry-run safety
- Tiny dashboard for health, candidates, audit, and top atoms

## Why this can beat typical agent memory stacks

| Capability | Typical RAG / Mem0-style | Ultimate Memory |
|---|---|---|
| Human-readable canon | Often opaque DB rows | Obsidian / Basic Memory markdown |
| Memory unit | Chunks or free-form notes | Typed atomic memories |
| Stale facts | Accumulate | Bi-temporal + auto-supersede |
| Ranking | Similarity only | RRF + salience + type priors |
| Local / private | Often cloud | Local-first, Docker optional |
| Agent interface | SDK / REST | MCP tools for Claude + Codex |

Fallback mode still works if Docker is down: Basic Memory + direct markdown + SQLite FTS + atoms.

## Quick Start

```powershell
cd "C:\Users\Ovindu\Documents\Pet Projects\ultimate-memory"
docker compose up -d
uv sync --extra dev
uv run ultimate-memory health
uv run ultimate-memory index-vault
uv run ultimate-memory-router
```

If Docker Desktop is not running, the router still starts and falls back to Basic Memory plus direct markdown search / local atoms.

## MCP Tools

- `memory_bootstrap(task, project_path?)` — type-aware context packet (prefs/procedures first)
- `memory_search(query, project_path?, memory_types?, limit?)` — hybrid RRF + salience
- `memory_ingest_log(client, session_id, project_path?, transcript_or_path, tags?)`
- `memory_reflect(source_refs, summary, facts, preferences, decisions, procedures, open_questions?)`
- `memory_atoms(query?, memory_types?, project_path?, limit?, include_superseded?)`
- `memory_consolidate(dry_run?)` — merge near-duplicate atoms (`dry_run=true` by default)
- `memory_graph_query(entity_or_topic, depth?)`
- `memory_promote(candidate_id)`
- `memory_supersede(old_ref, new_fact, reason, source_refs)`

## Client Wiring

The intended MCP command is:

```powershell
uv run --project "C:\Users\Ovindu\Documents\Pet Projects\ultimate-memory" ultimate-memory-router
```

Codex and Claude should call `memory_bootstrap` at session start, `memory_search` / `memory_atoms` as needed, and `memory_reflect` at session end.

Claude also has a `SessionEnd` hook wired to `scripts/ingest-claude-session.ps1`, which ingests the full transcript path into the private local log store (and auto-reflects when high-signal phrases are present).

## CLI extras

```bash
uv run ultimate-memory atoms --query "redis"
uv run ultimate-memory consolidate --dry-run
uv run ultimate-memory consolidate --no-dry-run
```

## Docker Router Mode

Claude and Codex use stdio on the Windows host. Docker mode is available for service-style clients:

```powershell
docker compose --profile router up -d
```

The streamable HTTP MCP endpoint is `http://127.0.0.1:8788/mcp`.
