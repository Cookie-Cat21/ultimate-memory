from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .adapters.basic_memory import BasicMemoryAdapter
from .adapters.graph import GraphAdapter
from .adapters.vector import VectorAdapter
from .chunking import chunk_text
from .config import Settings, load_settings
from .models import AuditEvent, MemoryChunk, MemoryType, ReflectionPayload, SearchResult, safe_slug
from .store import LocalStore


class MemoryRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.store = LocalStore(
            self.settings.sqlite_dir,
            self.settings.audit_dir,
            self.settings.candidates_dir,
        )
        self.basic = BasicMemoryAdapter(
            self.settings.paths.basic_memory_vault,
            self.settings.basic_memory,
            self.settings.retrieval,
        )
        self.vector = VectorAdapter(self.settings.qdrant, self.settings.retrieval)
        self.graph = GraphAdapter(self.settings.neo4j)
        self._vector_ready_cache = False
        self._graph_ready_cache = False

    @property
    def _vector_ready(self) -> bool:
        if not self._vector_ready_cache:
            self._vector_ready_cache = self.vector.is_available()
        return self._vector_ready_cache

    @property
    def _graph_ready(self) -> bool:
        if not self._graph_ready_cache:
            self._graph_ready_cache = self.graph.is_available()
        return self._graph_ready_cache

    def refresh_health(self) -> None:
        """Re-probe Qdrant and Neo4j, updating the memoized availability flags."""
        self._vector_ready_cache = self.vector.is_available()
        self._graph_ready_cache = self.graph.is_available()

    def health(self) -> dict:
        return {
            "basic_memory": self.basic.is_available(),
            "qdrant": self._vector_ready,
            "neo4j": self._graph_ready,
            "private_store": str(self.settings.paths.private_store),
            "vault": str(self.settings.paths.basic_memory_vault),
            "mode": "full" if self._vector_ready and self._graph_ready else "fallback",
        }

    def index_vault(self) -> dict:
        chunks = self.basic.note_chunks()
        for chunk in chunks:
            self.store.upsert_chunk(
                chunk_id=chunk.id,
                source_path=chunk.source_path,
                title=chunk.title,
                memory_type=chunk.memory_type.value,
                text=chunk.text,
                metadata=chunk.metadata,
                created_at=chunk.created_at,
            )
        vector_count = self.vector.upsert_chunks(chunks) if self._vector_ready else 0
        graph_count = self.graph.upsert_chunks(chunks) if self._graph_ready else 0
        self.store.write_audit(
            AuditEvent(
                action="index_vault",
                payload={
                    "chunks": len(chunks),
                    "qdrant_chunks": vector_count,
                    "graph_chunks": graph_count,
                },
            )
        )
        return {"chunks": len(chunks), "qdrant_chunks": vector_count, "graph_chunks": graph_count}

    def search(
        self,
        query: str,
        project_path: str | None = None,
        memory_types: list[str] | None = None,
        limit: int | None = None,
    ) -> dict:
        actual_limit = limit or self.settings.retrieval.default_limit
        vector_results = self.vector.search(query, actual_limit) if self._vector_ready else []
        bm_results = self.basic.search(query, actual_limit, include_cli=False)
        keyword_rows = self.store.keyword_search(query, actual_limit)
        keyword_results = [
            SearchResult(
                id=row["id"],
                title=row["title"],
                text=row["text"],
                source_path=row["source_path"],
                memory_type=row["memory_type"],
                score=self._keyword_score(query, row["text"]),
                provenance={"source": "sqlite-fts"},
            )
            for row in keyword_rows
        ]
        graph_hits = self.graph.query(query, depth=1) if self._graph_ready else []

        results = self._rank_results(
            [*vector_results, *keyword_results, *bm_results],
            memory_types=memory_types,
            limit=actual_limit,
        )
        if project_path:
            for result in results:
                result.provenance["requested_project_path"] = project_path
        return {
            "query": query,
            "mode": self.health()["mode"],
            "results": [result.model_dump() for result in results],
            "graph_hits": graph_hits[:5],
        }

    def bootstrap(self, task: str, project_path: str | None = None) -> dict:
        self.refresh_health()
        queries = [task]
        if project_path:
            queries.append(Path(project_path).name)
        queries.extend(["Ovindu preferences", "procedures", "recent decisions"])

        collected: list[SearchResult] = []
        for query in queries:
            search_result = self.search(query, project_path=project_path, limit=5)
            collected.extend(SearchResult(**item) for item in search_result["results"])

        ranked = self._rank_results(collected, limit=12)
        packet = self._fit_budget(ranked, self.settings.retrieval.bootstrap_token_budget_chars)
        return {
            "task": task,
            "project_path": project_path,
            "health": self.health(),
            "instructions": [
                "Use this packet instead of asking Ovindu to re-explain known context.",
                "Call memory_search for details instead of loading the whole vault.",
                "Call memory_reflect after meaningful work to update durable memory.",
            ],
            "context_packet": packet,
        }

    def ingest_log(
        self,
        client: str,
        session_id: str,
        transcript_or_path: str,
        project_path: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        source_text, source_kind = self._read_transcript_or_path(transcript_or_path)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_client = safe_slug(client, "client")
        safe_session = safe_slug(session_id, "session")
        log_dir = self.settings.logs_dir / safe_client
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{timestamp}-{safe_session}.md"
        header = [
            "---",
            f"client: {client}",
            f"session_id: {session_id}",
            f"project_path: {project_path or ''}",
            f"source_kind: {source_kind}",
            f"created_at: {datetime.now(UTC).isoformat()}",
            f"tags: {', '.join(tags or [])}",
            "---",
            "",
        ]
        log_path.write_text("\n".join(header) + source_text, encoding="utf-8")

        chunks = [
            MemoryChunk(
                id=f"log:{safe_client}:{safe_session}:{idx}",
                text=chunk,
                source_path=str(log_path),
                title=f"{client} session {session_id}",
                memory_type=MemoryType.LOG,
                project_path=project_path,
                tags=tags or [],
                metadata={"client": client, "session_id": session_id, "chunk_index": idx},
            )
            for idx, chunk in enumerate(
                chunk_text(
                    source_text,
                    self.settings.retrieval.chunk_chars,
                    self.settings.retrieval.chunk_overlap,
                )
            )
        ]
        for chunk in chunks:
            self.store.upsert_chunk(
                chunk_id=chunk.id,
                source_path=chunk.source_path,
                title=chunk.title,
                memory_type=chunk.memory_type.value,
                text=chunk.text,
                metadata=chunk.metadata,
                created_at=chunk.created_at,
            )
        vector_count = self.vector.upsert_chunks(chunks) if self._vector_ready else 0
        graph_count = self.graph.upsert_chunks(chunks) if self._graph_ready else 0
        self.store.write_audit(
            AuditEvent(
                action="ingest_log",
                payload={
                    "client": client,
                    "session_id": session_id,
                    "path": str(log_path),
                    "chunks": len(chunks),
                    "qdrant_chunks": vector_count,
                    "graph_chunks": graph_count,
                },
            )
        )
        return {
            "path": str(log_path),
            "chunks": len(chunks),
            "qdrant_chunks": vector_count,
            "graph_chunks": graph_count,
        }

    def reflect(self, payload: ReflectionPayload) -> dict:
        title = f"Memory Reflection {datetime.now(UTC).strftime('%Y-%m-%d %H%M%S')}"
        folder = "research/agent-memory/reflections"
        content = self._reflection_markdown(title, payload)
        duplicate_hits = self.search(payload.summary, limit=5)["results"]
        confidence = self._confidence(payload, duplicate_hits)
        should_write = confidence >= 0.55
        candidate_id = uuid4().hex

        result = {
            "candidate_id": candidate_id,
            "confidence": confidence,
            "canonical_written": False,
            "duplicate_hits": duplicate_hits[:3],
            "title": title,
            "folder": folder,
        }
        if should_write:
            self.basic.write_note(
                title=title,
                folder=folder,
                content=content,
                tags=["ultimate-memory", "reflection"],
            )
            if self._graph_ready:
                self.graph.add_reflection(
                    title=title,
                    memory_type="reflection",
                    text=payload.summary,
                    source_refs=payload.source_refs,
                )
            result["canonical_written"] = True
        else:
            self.store.write_candidate(
                candidate_id,
                {
                    "title": title,
                    "folder": folder,
                    "content": content,
                    "payload": payload.model_dump(),
                    "confidence": confidence,
                },
            )

        self.store.write_audit(
            AuditEvent(
                action="reflect",
                payload=result | {"reflection": payload.model_dump()},
                source_refs=payload.source_refs,
            )
        )
        return result

    def graph_query(self, entity_or_topic: str, depth: int = 1) -> dict:
        return {
            "entity_or_topic": entity_or_topic,
            "depth": depth,
            "available": self._graph_ready,
            "results": self.graph.query(entity_or_topic, depth) if self._graph_ready else [],
        }

    def promote(self, candidate_id: str) -> dict:
        candidate = self.store.read_candidate(candidate_id)
        if not candidate:
            return {"promoted": False, "reason": "candidate_not_found", "candidate_id": candidate_id}
        self.basic.write_note(
            title=candidate["title"],
            folder=candidate["folder"],
            content=candidate["content"],
            tags=["ultimate-memory", "promoted"],
        )
        self.store.delete_candidate(candidate_id)
        self.store.write_audit(
            AuditEvent(
                action="promote",
                payload={"candidate_id": candidate_id, "title": candidate["title"]},
            )
        )
        return {"promoted": True, "candidate_id": candidate_id, "title": candidate["title"]}

    def supersede(
        self,
        old_ref: str,
        new_fact: str,
        reason: str,
        source_refs: list[str] | None = None,
    ) -> dict:
        source_refs = source_refs or []
        title = f"Supersession {datetime.now(UTC).strftime('%Y-%m-%d %H%M%S')}"
        content = "\n".join(
            [
                f"# {title}",
                "",
                f"Old reference: {old_ref}",
                "",
                f"New fact: {new_fact}",
                "",
                f"Reason: {reason}",
                "",
                "## Source refs",
                *(f"- {ref}" for ref in source_refs),
                "",
                "## Relations",
                f"- supersedes [[{old_ref}]]",
            ]
        )
        self.basic.write_note(
            title=title,
            folder="research/agent-memory/supersessions",
            content=content,
            tags=["ultimate-memory", "supersession"],
        )
        graph_written = (
            self.graph.supersede(old_ref, new_fact, reason, source_refs) if self._graph_ready else False
        )
        self.store.write_audit(
            AuditEvent(
                action="supersede",
                payload={
                    "old_ref": old_ref,
                    "new_fact": new_fact,
                    "reason": reason,
                    "graph_written": graph_written,
                },
                source_refs=source_refs,
            )
        )
        return {"written": True, "graph_written": graph_written, "title": title}

    def dashboard_state(self) -> dict:
        return {
            "health": self.health(),
            "recent_audit": self.store.recent_audit(20),
            "candidates": self.store.list_candidates(),
            "indexed_sources": list(self.store.all_indexed_sources())[:100],
        }

    def _rank_results(
        self,
        results: list[SearchResult],
        memory_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        allowed = set(memory_types or [])
        filtered = [result for result in results if not allowed or result.memory_type in allowed]
        deduped: dict[str, SearchResult] = {}
        for result in filtered:
            key = result.source_path or result.id
            existing = deduped.get(key)
            if existing is None or result.score > existing.score:
                deduped[key] = result
        ranked = sorted(deduped.values(), key=lambda result: result.score, reverse=True)
        return ranked[: limit or self.settings.retrieval.default_limit]

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        haystack = text.lower()
        needle = query.lower().strip()
        terms = [term for term in needle.split() if len(term) > 1]
        term_hits = sum(1 for term in terms if term in haystack)
        score = 0.55 + min(term_hits * 0.06, 0.3)
        if needle and needle in haystack:
            score += 0.25
        return min(score, 1.2)

    @staticmethod
    def _fit_budget(results: list[SearchResult], budget_chars: int) -> list[dict]:
        packet: list[dict] = []
        used = 0
        for result in results:
            text = result.text.strip()
            remaining = budget_chars - used
            if remaining <= 0:
                break
            if len(text) > min(remaining, 1600):
                text = text[: min(remaining, 1600)].rstrip() + "..."
            item = result.model_dump()
            item["text"] = text
            packet.append(item)
            used += len(text)
        return packet

    @staticmethod
    def _read_transcript_or_path(transcript_or_path: str) -> tuple[str, str]:
        candidate = Path(transcript_or_path)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="ignore"), "path"
        return transcript_or_path, "inline"

    @staticmethod
    def _reflection_markdown(title: str, payload: ReflectionPayload) -> str:
        sections = [
            f"# {title}",
            "",
            payload.summary.strip(),
            "",
            "## Facts",
            *(f"- {item}" for item in payload.facts),
            "",
            "## Preferences",
            *(f"- {item}" for item in payload.preferences),
            "",
            "## Decisions",
            *(f"- {item}" for item in payload.decisions),
            "",
            "## Procedures",
            *(f"- {item}" for item in payload.procedures),
            "",
            "## Open Questions",
            *(f"- {item}" for item in payload.open_questions),
            "",
            "## Source Refs",
            *(f"- {item}" for item in payload.source_refs),
            "",
            "## Relations",
            "- relates_to [[Shared Claude Codex Agentic Memory Goal]]",
        ]
        return "\n".join(sections).strip() + "\n"

    @staticmethod
    def _confidence(payload: ReflectionPayload, duplicate_hits: list[dict]) -> float:
        signal_count = (
            len(payload.facts)
            + len(payload.preferences)
            + len(payload.decisions)
            + len(payload.procedures)
        )
        base = 0.4 + min(signal_count * 0.08, 0.4)
        if payload.source_refs:
            base += 0.1
        if duplicate_hits:
            base += 0.05
        return min(base, 0.95)
