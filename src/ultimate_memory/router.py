from __future__ import annotations

import concurrent.futures
import re
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

# Phrases that signal extractable knowledge in session transcripts.
_DECISION_SIGNALS = ("decided to", "decision:", "we chose", "going with", "will use", "switching to", "the approach is")
_PREFERENCE_SIGNALS = ("prefer ", "preference:", "always ", "never ", "i like", "i don't like", "i want", "please avoid")
_PROCEDURE_SIGNALS = ("steps:", "to do:", "procedure:", "how to ", "the process is", "to fix this")
_FACT_SIGNALS = ("the fix was", "the issue was", "root cause", "turns out", "note that", "important:", "the problem is", "bug:")


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
        tags: list[str] | None = None,
    ) -> dict:
        actual_limit = limit or self.settings.retrieval.default_limit

        # --- run all four sources in parallel --------------------------------
        def _vector():
            return self.vector.search(
                query, actual_limit, tags=tags, memory_types=memory_types
            ) if self._vector_ready else []

        def _keyword():
            rows = self.store.keyword_search(query, actual_limit)
            return [
                SearchResult(
                    id=row["id"],
                    title=row["title"],
                    text=row["text"],
                    source_path=row["source_path"],
                    memory_type=row["memory_type"],
                    score=self._keyword_score(query, row["text"]),
                    provenance={"source": "sqlite-fts"},
                )
                for row in rows
            ]

        def _basic():
            return self.basic.search(query, actual_limit, include_cli=False)

        def _graph():
            return self.graph.query(query, depth=1) if self._graph_ready else []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fv = pool.submit(_vector)
            fk = pool.submit(_keyword)
            fb = pool.submit(_basic)
            fg = pool.submit(_graph)
            vector_results: list[SearchResult] = fv.result()
            keyword_results: list[SearchResult] = fk.result()
            bm_results: list[SearchResult] = fb.result()
            graph_hits: list[dict] = fg.result()

        results = self._rrf_rank(
            [vector_results, keyword_results, bm_results],
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

        ranked = self._rrf_rank([collected], limit=12)
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

        # --- auto-extract learnings and reflect them into durable memory -----
        reflection_result: dict | None = None
        reflection_payload = self._extract_reflection(source_text, session_id, project_path)
        if reflection_payload is not None:
            try:
                reflection_result = self.reflect(reflection_payload, project_path=project_path)
            except Exception:
                pass

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
                    "auto_reflected": reflection_result is not None,
                },
            )
        )
        return {
            "path": str(log_path),
            "chunks": len(chunks),
            "qdrant_chunks": vector_count,
            "graph_chunks": graph_count,
            "reflection": reflection_result,
        }

    def reflect(self, payload: ReflectionPayload, project_path: str | None = None) -> dict:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H%M%S")
        title = self._smart_title(payload, timestamp)
        folder = "research/agent-memory/reflections"

        # Discover vault entities to auto-link
        note_titles = [p.stem for p in self.basic.iter_markdown_files()]
        reflection_text = " ".join(
            payload.facts + payload.decisions + payload.preferences + payload.procedures
        )
        entity_links = self._entity_links(reflection_text, note_titles)

        content = self._reflection_markdown(title, payload, project_path=project_path, entity_links=entity_links)
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

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rrf_rank(
        self,
        result_lists: list[list[SearchResult]],
        memory_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion across multiple independently-ranked result lists.

        Each list is treated as an independent ranking signal.  A result that
        appears highly in multiple lists gets a boosted fused score.  Scores are
        normalised to [0, 1] before returning so they stay comparable regardless
        of how many lists contributed.
        """
        k = 60  # standard RRF constant
        allowed = set(memory_types or [])

        rrf_scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                if allowed and result.memory_type not in allowed:
                    continue
                key = result.source_path or result.id
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                if key not in best_result or result.score > best_result[key].score:
                    best_result[key] = result

        # Normalise to [0, 1]
        if rrf_scores:
            max_score = max(rrf_scores.values())
            if max_score > 0:
                for key in rrf_scores:
                    rrf_scores[key] /= max_score

        fused: list[SearchResult] = []
        for key, result in best_result.items():
            result.provenance["raw_score"] = result.score
            result.provenance["rrf_score"] = rrf_scores[key]
            result.score = rrf_scores[key]
            fused.append(result)

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[: limit or self.settings.retrieval.default_limit]

    # ------------------------------------------------------------------
    # Smart extraction from session transcripts
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reflection(
        text: str,
        session_id: str,
        project_path: str | None,
    ) -> ReflectionPayload | None:
        """Parse a transcript for high-signal phrases and build a ReflectionPayload.

        Returns None when the transcript doesn't contain enough signal to be
        worth auto-reflecting (avoids polluting Obsidian with noise).
        """
        facts: list[str] = []
        decisions: list[str] = []
        preferences: list[str] = []
        procedures: list[str] = []

        seen: set[str] = set()

        def _add(bucket: list[str], line: str) -> None:
            norm = re.sub(r"\s+", " ", line.strip())[:220]
            if norm and norm not in seen and len(norm) >= 20:
                seen.add(norm)
                bucket.append(norm)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            lower = line.lower()

            if any(sig in lower for sig in _DECISION_SIGNALS):
                _add(decisions, line)
            elif any(sig in lower for sig in _PREFERENCE_SIGNALS):
                _add(preferences, line)
            elif any(sig in lower for sig in _PROCEDURE_SIGNALS):
                _add(procedures, line)
            elif any(sig in lower for sig in _FACT_SIGNALS):
                _add(facts, line)

        # Cap per category to avoid runaway reflections
        facts = facts[:6]
        decisions = decisions[:6]
        preferences = preferences[:4]
        procedures = procedures[:4]

        total_signals = len(facts) + len(decisions) + len(preferences) + len(procedures)
        if total_signals < 1:
            return None

        project_name = Path(project_path).name if project_path else "session"
        summary = (
            f"Auto-extracted from {project_name} session {session_id[:20]}. "
            f"{total_signals} signals: {len(decisions)} decisions, {len(facts)} facts, "
            f"{len(preferences)} preferences, {len(procedures)} procedures."
        )
        return ReflectionPayload(
            summary=summary,
            facts=facts,
            decisions=decisions,
            preferences=preferences,
            procedures=procedures,
            source_refs=[f"session:{session_id}"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        haystack = text.lower()
        needle = query.lower().strip()
        terms = [term for term in needle.split() if len(term) > 1]
        term_hits = sum(1 for term in terms if term in haystack)
        score = 0.55 + min(term_hits * 0.06, 0.3)
        if needle and needle in haystack:
            score += 0.25
        return min(score, 1.0)

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
    def _smart_title(payload: ReflectionPayload, timestamp: str) -> str:
        """Derive a meaningful, filesystem-safe title from the first decision or fact."""
        first_signal = (
            next(iter(payload.decisions), None)
            or next(iter(payload.facts), None)
            or next(iter(payload.preferences), None)
        )
        if not first_signal:
            return f"Memory Reflection {timestamp}"
        # Strip role prefixes like "Assistant: " or "User: "
        clean = re.sub(r"^(assistant|user|system)\s*:\s*", "", first_signal, flags=re.IGNORECASE)
        truncated = clean[:72].rstrip()
        # Remove chars illegal in Windows/macOS filenames
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", truncated).strip("-").strip()
        return safe or f"Memory Reflection {timestamp}"

    @staticmethod
    def _entity_links(text: str, note_titles: list[str]) -> list[str]:
        """Return vault note titles that are mentioned in *text* (case-insensitive).

        Skips very short or generic titles to avoid false matches.
        """
        text_lower = text.lower()
        matched: list[str] = []
        for title in note_titles:
            if len(title) < 4:
                continue
            # Skip timestamped reflection notes — they're not useful link targets
            if re.match(r"Memory Reflection \d", title):
                continue
            if title.lower() in text_lower:
                matched.append(title)
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in matched:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:8]  # cap to keep Relations section readable

    @staticmethod
    def _reflection_markdown(
        title: str,
        payload: ReflectionPayload,
        project_path: str | None = None,
        entity_links: list[str] | None = None,
    ) -> str:
        relations = ["- relates_to [[Shared Claude Codex Agentic Memory Goal]]"]
        if project_path:
            project_name = Path(project_path).name
            if project_name:
                relations.append(f"- from_project [[{project_name}]]")
        for entity in (entity_links or []):
            relations.append(f"- mentions [[{entity}]]")

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
            *relations,
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
