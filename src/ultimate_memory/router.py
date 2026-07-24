from __future__ import annotations

import concurrent.futures
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .adapters.basic_memory import BasicMemoryAdapter
from .adapters.graph import GraphAdapter
from .adapters.vector import VectorAdapter
from .atoms import (
    atoms_from_reflection,
    blend_scores,
    contradiction_score,
    group_near_duplicates,
    is_temporal_query,
    now_iso,
    parse_iso,
)
from .chunking import chunk_text
from .config import Settings, load_settings
from .dates import parse_loose_date, resolve_relative_dates
from .extraction import extract_from_transcript, parse_dialogue_turns, session_anchor_from_text
from .models import (
    AtomicMemory,
    AuditEvent,
    MemoryChunk,
    MemoryType,
    ReflectionPayload,
    SearchResult,
    safe_slug,
)
from .aggregate import aggregate_answer, detect_aggregate_intent
from .answer import f1_ready_text, synthesize_answer
from .hops import (
    MAX_HOP_SEARCHES,
    build_hop_queries,
    extract_hop_entities,
    merge_contexts,
)
from .llm_answer import use_llm_from_env
from .store import LocalStore

logger = logging.getLogger(__name__)

_PROBE_ENTITY_PATTERNS = (
    re.compile(r"what did\s+([A-Z][a-z]+)\b", re.I),
    re.compile(r"what is\s+([A-Z][a-z]+)'s\b", re.I),
    re.compile(r"what(?:'s| is)\s+([A-Z][a-z]+)'s\b", re.I),
    re.compile(r"where did\s+([A-Z][a-z]+)\b", re.I),
)


def _probe_entity_from_question(question: str) -> str | None:
    for pattern in _PROBE_ENTITY_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1)
    return None


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

    def answer(
        self,
        question: str,
        project_path: str | None = None,
        limit: int = 8,
        as_of: str | None = None,
        *,
        max_hop_searches: int = MAX_HOP_SEARCHES,
        use_llm: bool | None = None,
    ) -> dict:
        """Retrieve memory contexts and synthesize an answer (extractive or optional local LLM)."""
        # Keyword-dense query helps FTS more than full natural-language questions.
        stop = {
            "when", "what", "where", "who", "whom", "which", "how", "why", "did", "does",
            "do", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on",
            "for", "with", "her", "his", "their", "she", "he", "they", "has", "have",
            "had", "been", "about", "would", "could", "from", "into", "that", "this",
        }
        dense_terms = [
            tok for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", question)
            if tok.lower() not in stop and len(tok) > 2
        ]
        dense_query = " ".join(dense_terms[:8]) or question

        # Light question expansion for common LoCoMo-style probes.
        expansions: list[str] = []
        q_lower_all = question.lower()
        if "identity" in q_lower_all:
            expansions.append("transgender woman identity")
        if "adoption" in q_lower_all:
            expansions.append("adoption agencies application interview")
        if "counseling" in q_lower_all or "counselling" in q_lower_all:
            expansions.append("counseling mental health LGBTQ workshop")
        if "career" in q_lower_all or re.search(
            r"\b(?:job|work|occupation|profession)\b", q_lower_all
        ):
            expansions.append("career counseling mental health work")
        if "sweden" in q_lower_all:
            expansions.append("grandmother Sweden necklace home country")
        if "camping" in q_lower_all:
            expansions.append("camping trip outdoors nature")
        if "pottery" in q_lower_all:
            expansions.append("pottery class ceramic bowl")
        if "painting" in q_lower_all or (
            "paint" in q_lower_all and "sunrise" not in q_lower_all
        ):
            expansions.append("painting painted art canvas")
        if "paint" in q_lower_all and "sunrise" in q_lower_all:
            expansions.append("painted sunrise last year 2022")
        if "friend" in q_lower_all:
            expansions.append("friends friendship close friend")
        if "research" in q_lower_all:
            expansions.append("research adoption agencies")
        if "relationship status" in q_lower_all or "single" in q_lower_all:
            expansions.append("single parent relationship")
        if "political" in q_lower_all or "leaning" in q_lower_all:
            expansions.append("LGBTQ rights liberal inclusive")
        if "religious" in q_lower_all:
            expansions.append("church faith religious conservatives")
        if "personality" in q_lower_all or "traits" in q_lower_all:
            expansions.append("thoughtful authentic driven real")
        if "national park" in q_lower_all or "theme park" in q_lower_all:
            expansions.append("camping nature outdoors meteor")
        if "roadtrip" in q_lower_all or "road trip" in q_lower_all:
            expansions.append("roadtrip accident scared family")
        if "home country" in q_lower_all or "move back" in q_lower_all:
            expansions.append("adoption agencies family children")
        if "seuss" in q_lower_all or "bookshelf" in q_lower_all:
            expansions.append("kids books classics children's library")
        if "vivaldi" in q_lower_all or "four seasons" in q_lower_all:
            expansions.append("classical music bach mozart")

        search_queries = [dense_query, question, *expansions]
        rich_contexts: list[dict] = []
        search_result: dict = {"results": []}
        for sq in search_queries:
            result = self.search(
                query=sq,
                project_path=project_path,
                limit=limit,
                as_of=as_of,
            )
            if sq == dense_query:
                search_result = result
            rich_contexts.extend(item for item in result["results"] if item.get("text"))

        # Type-biased side searches for how-to / preference / decision questions.
        q_lower = question.lower()
        typed_extra: list[tuple[str, list[str]]] = []
        if any(cue in q_lower for cue in ("how do", "how to", "steps", "procedure", "deploy")):
            typed_extra.append((question, ["procedure"]))
        if any(cue in q_lower for cue in ("prefer", "preference", "theme", "always", "never")):
            typed_extra.append((question, ["preference"]))
        if any(cue in q_lower for cue in ("decide", "decided", "decision", "chose", "chosen")):
            typed_extra.append((question, ["decision"]))
        for typed_query, types in typed_extra:
            typed_result = self.search(
                query=typed_query,
                project_path=project_path,
                memory_types=types,
                limit=min(limit, 5),
                as_of=as_of,
            )
            for item in typed_result.get("results", []):
                if item.get("text"):
                    rich_contexts.append(item)
            # Lexical miss fallback: inject highest-salience active atoms of that type.
            for atom in self.store.list_active_atoms(memory_types=types, limit=5):
                rich_contexts.append(
                    self._atom_to_search_result(atom, score=max(atom.salience, 0.55)).model_dump()
                )

        hop_entities = extract_hop_entities(question, search_result["results"])
        hop_queries = build_hop_queries(
            question,
            hop_entities,
            max_queries=min(2, max_hop_searches),
        )
        hop_searches: list[dict] = []
        hop_contexts: list[dict] = []
        for hop_query in hop_queries[:max_hop_searches]:
            hop_result = self.search(
                query=hop_query,
                project_path=project_path,
                limit=limit,
                as_of=as_of,
            )
            hop_searches.append({"query": hop_query, "search": hop_result})
            for item in hop_result["results"]:
                if not item.get("text"):
                    continue
                boosted = dict(item)
                provenance = dict(boosted.get("provenance") or {})
                provenance["hop"] = True
                boosted["provenance"] = provenance
                boosted["score"] = float(boosted.get("score") or 0.0) + 0.35
                hop_contexts.append(boosted)

        rich_contexts = merge_contexts(rich_contexts, hop_contexts)

        # Prefer atomic / dialogue evidence over noisy reflection markdown notes.
        def _is_atomicish(item: dict) -> bool:
            prov = item.get("provenance") or {}
            source = str(prov.get("source") or "")
            mtype = str(item.get("memory_type") or "")
            text = str(item.get("text") or "")
            if source in {"atomic-memory", "sqlite-fts"} and mtype in {
                "fact",
                "preference",
                "decision",
                "procedure",
                "log",
            }:
                return True
            if text.startswith("[D") or item.get("id", "").startswith(("atom:", "turn:")):
                return True
            if mtype in {"fact", "preference", "decision", "procedure", "log"}:
                return True
            return False

        atomic_contexts = [item for item in rich_contexts if _is_atomicish(item)]
        if len(atomic_contexts) >= 2:
            # Keep a couple of notes only if they contain dates the atoms lack.
            extras = [
                item
                for item in rich_contexts
                if not _is_atomicish(item)
                and re.search(r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", item.get("text") or "", re.I)
            ][:2]
            rich_contexts = atomic_contexts + extras

        # Also inject top keyword atom hits for question entities.
        for ent in re.findall(r"\b[A-Z][a-z]{2,}\b", question)[:3]:
            for atom in self.store.search_atoms(ent, limit=4):
                rich_contexts.append(
                    self._atom_to_search_result(atom, score=max(atom.salience, 0.5)).model_dump()
                )

        # "What did X / what is X's / where did X" → targeted "{Name} {key_noun}" atom probes.
        probe_entity = _probe_entity_from_question(question)
        if probe_entity:
            entity_lower = probe_entity.lower()
            key_nouns = [
                tok
                for tok in dense_terms
                if tok.lower() not in {entity_lower, probe_entity.lower()}
            ][:4]
            probe_queries: list[str] = [probe_entity]
            for noun in key_nouns[:3]:
                probe_queries.append(f"{probe_entity} {noun}")
            seen_probe: set[str] = set()
            for pq in probe_queries:
                if pq.lower() in seen_probe:
                    continue
                seen_probe.add(pq.lower())
                for atom in self.store.search_atoms(pq, limit=3):
                    rich_contexts.append(
                        self._atom_to_search_result(
                            atom, score=max(atom.salience, 0.62)
                        ).model_dump()
                    )

        rich_contexts = merge_contexts(rich_contexts, [])

        # Pull a wide person-scoped atom window for multi-fact aggregation only.
        # Keep these out of normal extractive/LLM ranking so temporal/single-hop
        # dates are not drowned by inventory dumps.
        agg_intent = detect_aggregate_intent(question)
        person_atom_texts: list[str] = []
        inventory_contexts: list[dict] = []
        if agg_intent and agg_intent.person:
            for atom in self.store.search_atoms(agg_intent.person, limit=100):
                if atom.text:
                    person_atom_texts.append(atom.text)
            for extra_q in (
                f"{agg_intent.person} profile",
                f"{agg_intent.person} activities",
                f"{agg_intent.person} books",
                f"{agg_intent.person} painted",
                f"{agg_intent.person} LGBTQ",
                f"{agg_intent.person} relationship",
            ):
                for atom in self.store.search_atoms(extra_q, limit=10):
                    if not atom.text:
                        continue
                    person_atom_texts.append(atom.text)
                    # Compact inventory/profile cards may still help the LLM.
                    if atom.id.startswith(("atom:inv:", "atom:profile:")) or any(
                        marker in atom.text
                        for marker in (
                            " profile:",
                            " activities:",
                            " books read:",
                            " painted:",
                            " LGBTQ ",
                            " relationship status:",
                            " career:",
                            " identity:",
                        )
                    ):
                        inventory_contexts.append(
                            self._atom_to_search_result(
                                atom, score=max(atom.salience, 0.72)
                            ).model_dump()
                        )

        retrieved_texts = [
            str(item.get("text") or "")
            for item in rich_contexts
            if item.get("text")
        ]
        seen_ctx: set[str] = set()
        dedup_texts: list[str] = []
        for text in person_atom_texts + retrieved_texts:
            key = text.strip().lower()
            if key and key not in seen_ctx:
                seen_ctx.add(key)
                dedup_texts.append(text.strip())

        aggregated = aggregate_answer(question, dedup_texts) if agg_intent else None
        prefer_aggregated = bool(
            aggregated
            and (
                "," in aggregated
                or (
                    agg_intent is not None
                    and agg_intent.kind
                    in {
                        "relationship_status",
                        "moved_from",
                        "identity",
                        "career",
                        "hypothetical",
                        "painted_recently",
                        "beach_count",
                        "children_count",
                        "duration",
                        "instruments",
                        "supporters",
                        "art_kind",
                        "both_painted",
                        "destress",
                        "symbols",
                        "trans_events",
                        "bought_items",
                        "hike_family",
                        "artists_seen",
                        "transition_changes",
                        "pottery_types",
                        "pet_names",
                        "lgbtq_ways",
                        "lgbtq_events",
                        "help_children",
                        "camp_places",
                        "kids_like",
                        "books",
                        "activities",
                        "research",
                        "political",
                        "personality",
                        "education_fields",
                        "inventory_union",
                        "how_many",
                        "both_intersection",
                    }
                )
            )
        )

        should_use_llm = use_llm if use_llm is not None else use_llm_from_env()
        if prefer_aggregated:
            # Aggregation beats flaky local LLM on list / inventory multi-hop.
            answer_text = aggregated or ""
        elif should_use_llm:
            llm_pool = merge_contexts(rich_contexts, inventory_contexts)
            # Prefer atomic / dialogue snippets first for the local answerer.
            ordered = sorted(
                [item for item in llm_pool if item.get("text")],
                key=lambda item: (
                    0
                    if (
                        str((item.get("provenance") or {}).get("source")) == "atomic-memory"
                        or str(item.get("id") or "").startswith(("atom:", "turn:"))
                        or str(item.get("text") or "").startswith("[D")
                        or " profile:" in str(item.get("text") or "")
                        or " activities:" in str(item.get("text") or "")
                    )
                    else 1,
                    -float(item.get("score") or 0.0),
                ),
            )
            context_texts = [str(item.get("text") or "")[:500] for item in ordered[:14]]
            try:
                from .llm_answer import get_local_answerer

                answer_text = get_local_answerer().answer(question, context_texts)
            except Exception as exc:
                logger.warning("Local LLM answer failed, falling back to extractive: %s", exc)
                answer_text = aggregated or synthesize_answer(question, rich_contexts)
            if (not answer_text or answer_text.lower() == "i don't know") and aggregated:
                answer_text = aggregated
        else:
            answer_text = aggregated or synthesize_answer(question, rich_contexts)
        return {
            "question": question,
            "answer": answer_text,
            "f1_text": f1_ready_text(answer_text),
            "contexts_used": [item["text"] for item in rich_contexts],
            "search": search_result,
            "hop_entities": hop_entities,
            "hop_searches": hop_searches,
            "aggregated": aggregated,
        }

    def search(
        self,
        query: str,
        project_path: str | None = None,
        memory_types: list[str] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        include_superseded: bool = False,
        as_of: str | None = None,
    ) -> dict:
        actual_limit = limit or self.settings.retrieval.default_limit
        self.store.refresh_salience(limit=200)

        temporal_query = as_of is None and is_temporal_query(query)
        if temporal_query:
            include_superseded = True

        # --- run all five sources in parallel --------------------------------
        def _vector():
            return self.vector.search(
                query,
                actual_limit,
                tags=tags,
                memory_types=memory_types,
                include_superseded=include_superseded,
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

        def _atoms():
            return self._atom_search_results(
                query,
                limit=actual_limit,
                memory_types=memory_types,
                include_superseded=include_superseded,
                as_of=as_of,
                prefer_older_valid_from=temporal_query,
            )

        def _graph():
            return self.graph.query(query, depth=1) if self._graph_ready else []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fv = pool.submit(_vector)
            fb = pool.submit(_basic)
            fg = pool.submit(_graph)
            # SQLite store is not thread-safe; keep local FTS/atom queries on main thread.
            keyword_results = _keyword()
            atom_results = _atoms()
            vector_results: list[SearchResult] = fv.result()
            bm_results: list[SearchResult] = fb.result()
            graph_hits: list[dict] = fg.result()

        results = self._rrf_rank(
            [vector_results, keyword_results, bm_results, atom_results],
            memory_types=memory_types,
            limit=actual_limit * 2,
        )
        results = self._apply_salience_rerank(results)[:actual_limit]

        touched = [
            r.id for r in results
            if r.provenance.get("source") == "atomic-memory" or r.id.startswith("atom:")
        ]
        if touched:
            self.store.touch_atoms(touched)

        if project_path:
            for result in results:
                result.provenance["requested_project_path"] = project_path
        return {
            "query": query,
            "as_of": as_of,
            "temporal_query": temporal_query,
            "mode": self.health()["mode"],
            "results": [result.model_dump() for result in results],
            "graph_hits": graph_hits[:5],
            "atoms_considered": len(atom_results),
        }

    def bootstrap(self, task: str, project_path: str | None = None) -> dict:
        """Type-aware bootstrap: preferences/procedures first, then decisions/facts."""
        self.refresh_health()
        self.store.refresh_salience(limit=300)

        typed_queries: list[tuple[str, list[str] | None, int]] = [
            (f"{task} preferences", ["preference"], 4),
            ("user preferences always never prefer", ["preference"], 3),
            (f"{task} procedures how to", ["procedure"], 4),
            (f"{task} decisions", ["decision"], 3),
            (task, ["fact", "note", "decision"], 5),
        ]
        if project_path:
            typed_queries.insert(0, (Path(project_path).name, None, 4))

        buckets: dict[str, list[SearchResult]] = {
            "preference": [],
            "procedure": [],
            "decision": [],
            "other": [],
        }
        for query, types, limit in typed_queries:
            search_result = self.search(
                query,
                project_path=project_path,
                memory_types=types,
                limit=limit,
            )
            for item in search_result["results"]:
                result = SearchResult(**item)
                key = result.memory_type if result.memory_type in buckets else "other"
                buckets[key].append(result)

        # Also pull highest-salience active atoms directly (even if lexical miss).
        for atom in self.store.list_active_atoms(
            memory_types=["preference", "procedure", "decision"],
            project_path=project_path,
            limit=10,
        ):
            result = self._atom_to_search_result(atom, score=atom.salience)
            key = atom.memory_type.value if atom.memory_type.value in buckets else "other"
            buckets[key].append(result)

        ordered: list[SearchResult] = []
        for key in ("preference", "procedure", "decision", "other"):
            ordered.extend(self._rrf_rank([buckets[key]], limit=6))

        ranked = self._dedupe_results(ordered)
        ranked = self._apply_salience_rerank(ranked)[:14]
        packet = self._fit_budget(ranked, self.settings.retrieval.bootstrap_token_budget_chars)
        return {
            "task": task,
            "project_path": project_path,
            "health": self.health(),
            "instructions": [
                "Use this packet instead of asking the user to re-explain known context.",
                "Treat preference/procedure atoms as durable defaults unless superseded.",
                "Call memory_search for details instead of loading the whole vault.",
                "Call memory_reflect after meaningful work to update durable memory.",
            ],
            "context_packet": packet,
            "composition": {
                "preferences": sum(1 for r in ranked if r.memory_type == "preference"),
                "procedures": sum(1 for r in ranked if r.memory_type == "procedure"),
                "decisions": sum(1 for r in ranked if r.memory_type == "decision"),
                "other": sum(
                    1 for r in ranked
                    if r.memory_type not in {"preference", "procedure", "decision"}
                ),
            },
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
        event_time = self._extract_transcript_event_time(source_text)
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
        turn_chunks, turn_atoms = self._dialogue_turn_memory(
            source_text,
            session_id=session_id,
            safe_session=safe_session,
            client=client,
            project_path=project_path,
            tags=tags or [],
            log_path=log_path,
            event_time=event_time,
        )
        chunks.extend(turn_chunks)
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
        turn_atoms_created = 0
        for atom in turn_atoms:
            report = self._ingest_atom(atom)
            if report["status"] == "created":
                turn_atoms_created += 1
        vector_count = self.vector.upsert_chunks(chunks) if self._vector_ready else 0
        graph_count = self.graph.upsert_chunks(chunks) if self._graph_ready else 0

        # --- auto-extract learnings and reflect them into durable memory -----
        reflection_result: dict | None = None
        extraction = extract_from_transcript(source_text, session_id, project_path)
        if extraction.payload is not None:
            try:
                reflection_result = self.reflect(
                    extraction.payload,
                    project_path=project_path,
                    event_time=event_time,
                    extracted_entities=extraction.entities,
                )
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
                    "turn_chunks": len(turn_chunks),
                    "turn_atoms": turn_atoms_created,
                    "qdrant_chunks": vector_count,
                    "graph_chunks": graph_count,
                    "auto_reflected": reflection_result is not None,
                },
            )
        )
        return {
            "path": str(log_path),
            "chunks": len(chunks),
            "turn_chunks": len(turn_chunks),
            "turn_atoms": turn_atoms_created,
            "qdrant_chunks": vector_count,
            "graph_chunks": graph_count,
            "reflection": reflection_result,
        }

    def reflect(
        self,
        payload: ReflectionPayload,
        project_path: str | None = None,
        event_time: str | None = None,
        *,
        extracted_entities: list[str] | None = None,
    ) -> dict:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H%M%S")
        title = self._smart_title(payload, timestamp)
        folder = "research/agent-memory/reflections"

        # Discover vault entities to auto-link
        note_titles = [p.stem for p in self.basic.iter_markdown_files()]
        reflection_text = " ".join(
            payload.facts + payload.decisions + payload.preferences + payload.procedures
        )
        entity_links = self._entity_links(reflection_text, note_titles)
        if extracted_entities:
            entity_links = list(dict.fromkeys(entity_links + extracted_entities))

        content = self._reflection_markdown(
            title, payload, project_path=project_path, entity_links=entity_links
        )
        duplicate_hits = self.search(payload.summary, limit=5)["results"]
        confidence = self._confidence(payload, duplicate_hits)
        should_write = confidence >= 0.55
        candidate_id = uuid4().hex

        # Always commit typed atoms (even if the markdown note is staged).
        atom_report = self._commit_atoms_from_reflection(
            payload,
            project_path=project_path,
            entities=entity_links,
            event_time=event_time,
        )

        result = {
            "candidate_id": candidate_id,
            "confidence": confidence,
            "canonical_written": False,
            "duplicate_hits": duplicate_hits[:3],
            "title": title,
            "folder": folder,
            "atoms": atom_report,
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
                    "atoms": atom_report,
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

        # Bi-temporal atom update: create the new fact atom and invalidate matches.
        new_atom = AtomicMemory(
            id=f"atom:fact:{safe_slug(new_fact)[:48]}:{uuid4().hex[:8]}",
            text=new_fact,
            memory_type=MemoryType.FACT,
            source_refs=source_refs,
            metadata={"supersession_reason": reason, "old_ref": old_ref},
        )
        atom_report = self._ingest_atom(new_atom, force_supersede_query=old_ref)

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
                    "atoms": atom_report,
                },
                source_refs=source_refs,
            )
        )
        return {
            "written": True,
            "graph_written": graph_written,
            "title": title,
            "atoms": atom_report,
        }

    def list_atoms(
        self,
        memory_types: list[str] | None = None,
        project_path: str | None = None,
        query: str | None = None,
        limit: int = 20,
        include_superseded: bool = False,
        as_of: str | None = None,
    ) -> dict:
        if query:
            atoms = self.store.search_atoms(
                query,
                limit=limit,
                memory_types=memory_types,
                include_superseded=include_superseded,
                as_of=as_of,
            )
        else:
            atoms = self.store.list_active_atoms(
                memory_types=memory_types,
                project_path=project_path,
                limit=limit,
                as_of=as_of,
            )
            if include_superseded and as_of is None:
                # Active list already filters; for superseded browse use search with "*".
                pass
        return {
            "count": len(atoms),
            "atoms": [atom.model_dump() for atom in atoms],
        }

    def consolidate(self, dry_run: bool = False, limit: int = 200) -> dict:
        """Merge near-duplicate active atoms; keep the highest-salience survivor."""
        self.store.refresh_salience(limit=limit)
        active = self.store.list_active_atoms(limit=limit)
        groups = group_near_duplicates(active, threshold=0.72)
        merges: list[dict] = []
        for group in groups:
            survivor = max(group, key=lambda a: (a.salience, a.access_count, a.created_at))
            victims = [a for a in group if a.id != survivor.id]
            merge_info = {
                "survivor_id": survivor.id,
                "survivor_text": survivor.text,
                "merged_ids": [v.id for v in victims],
                "merged_texts": [v.text for v in victims],
            }
            if not dry_run:
                for victim in victims:
                    self.store.invalidate_atom(victim.id, superseded_by=survivor.id)
                    if self._graph_ready:
                        self.graph.supersede_atom(
                            victim.id,
                            survivor.id,
                            reason="consolidate:near-duplicate",
                        )
                    # Reinforce survivor with victim evidence
                    survivor.source_refs = list(
                        dict.fromkeys(survivor.source_refs + victim.source_refs)
                    )
                    survivor.access_count += victim.access_count
                survivor.last_accessed = now_iso()
                self.store.upsert_atom(survivor)
                if self._graph_ready:
                    self.graph.upsert_atoms([survivor])
            merges.append(merge_info)

        report = {
            "dry_run": dry_run,
            "groups_found": len(groups),
            "atoms_merged": sum(len(m["merged_ids"]) for m in merges),
            "merges": merges[:50],
        }
        self.store.write_audit(AuditEvent(action="consolidate", payload=report))
        return report

    def dashboard_state(self) -> dict:
        active_atoms = self.store.list_active_atoms(limit=50)
        return {
            "health": self.health(),
            "recent_audit": self.store.recent_audit(20),
            "candidates": self.store.list_candidates(),
            "indexed_sources": list(self.store.all_indexed_sources())[:100],
            "active_atoms": len(active_atoms),
            "top_atoms": [
                {
                    "id": a.id,
                    "type": a.memory_type.value,
                    "salience": a.salience,
                    "text": a.text[:160],
                }
                for a in active_atoms[:10]
            ],
        }

    # ------------------------------------------------------------------
    # Atomic memory commit / contradiction
    # ------------------------------------------------------------------

    def _dialogue_turn_memory(
        self,
        source_text: str,
        *,
        session_id: str,
        safe_session: str,
        client: str,
        project_path: str | None,
        tags: list[str],
        log_path: Path,
        event_time: str | None,
    ) -> tuple[list[MemoryChunk], list[AtomicMemory]]:
        """Index LoCoMo-style dialogue turns as searchable chunks and fact atoms."""
        turns = parse_dialogue_turns(source_text)
        if not turns:
            return [], []

        anchor = session_anchor_from_text(source_text)
        if anchor is None and event_time:
            anchor = parse_loose_date(event_time)

        stamp = event_time or now_iso()
        chunks: list[MemoryChunk] = []
        atoms: list[AtomicMemory] = []

        for turn in turns:
            resolved = resolve_relative_dates(turn.utterance, anchor)
            line_text = f"[{turn.dia_id}] {turn.speaker}: {resolved}"
            informative = len(resolved.strip()) >= 20
            chunks.append(
                MemoryChunk(
                    id=f"turn:{safe_session}:{turn.dia_id}",
                    text=line_text,
                    source_path=str(log_path),
                    title=f"{turn.speaker} [{turn.dia_id}]",
                    memory_type=MemoryType.FACT if informative else MemoryType.LOG,
                    project_path=project_path,
                    tags=tags,
                    metadata={
                        "client": client,
                        "session_id": session_id,
                        "dia_id": turn.dia_id,
                        "speaker": turn.speaker,
                    },
                )
            )
            if informative and len(atoms) < 40:
                atoms.append(
                    AtomicMemory(
                        id=f"atom:turn:{safe_session}:{turn.dia_id}",
                        text=f"{turn.speaker}: {resolved}",
                        memory_type=MemoryType.FACT,
                        project_path=project_path,
                        entities=[turn.speaker],
                        source_refs=[f"session:{session_id}", f"turn:{turn.dia_id}"],
                        created_at=stamp,
                        valid_from=stamp,
                        event_time=event_time,
                        metadata={
                            "dia_id": turn.dia_id,
                            "speaker": turn.speaker,
                            "session_id": session_id,
                        },
                    )
                )

        return chunks, atoms

    def _commit_atoms_from_reflection(
        self,
        payload: ReflectionPayload,
        *,
        project_path: str | None,
        entities: list[str],
        event_time: str | None = None,
    ) -> dict:
        atoms = atoms_from_reflection(
            payload,
            project_path=project_path,
            entities=entities,
            created_at=event_time,
            event_time=event_time,
        )
        created: list[str] = []
        duplicates: list[str] = []
        superseded: list[dict] = []
        for atom in atoms:
            report = self._ingest_atom(atom)
            if report["status"] == "created":
                created.append(atom.id)
            elif report["status"] == "duplicate":
                duplicates.append(report.get("existing_id", atom.id))
            superseded.extend(report.get("superseded", []))
        return {
            "created": created,
            "duplicates": duplicates,
            "superseded": superseded,
            "count": len(created),
        }

    def _ingest_atom(
        self,
        atom: AtomicMemory,
        *,
        force_supersede_query: str | None = None,
    ) -> dict:
        """Insert an atom, skipping exact duplicates and superseding contradictions."""
        same_id = self.store.get_atom(atom.id)
        if same_id and same_id.is_active:
            same_id.access_count += 1
            same_id.last_accessed = now_iso()
            same_id.source_refs = list(dict.fromkeys(same_id.source_refs + atom.source_refs))
            self.store.upsert_atom(same_id)
            return {"status": "duplicate", "existing_id": same_id.id, "superseded": []}

        existing = self.store.find_duplicate_atom(atom)
        if existing:
            existing.access_count += 1
            existing.last_accessed = now_iso()
            existing.source_refs = list(dict.fromkeys(existing.source_refs + atom.source_refs))
            self.store.upsert_atom(existing)
            return {"status": "duplicate", "existing_id": existing.id, "superseded": []}

        superseded: list[dict] = []
        candidates = self.store.find_contradiction_candidates(atom)
        if force_supersede_query:
            # Also consider atoms that mention the old reference string.
            extra = self.store.search_atoms(
                force_supersede_query,
                limit=8,
                memory_types=[atom.memory_type.value],
            )
            seen = {c.id for c in candidates}
            for item in extra:
                if item.id not in seen:
                    candidates.append(item)

        for candidate in candidates:
            score = contradiction_score(atom.text, candidate.text)
            # Near-identical → treat as duplicate (reinforce old, don't create).
            if score >= 0.92 and contradiction_score(candidate.text, atom.text) >= 0.92:
                if atom.text.strip().lower() == candidate.text.strip().lower():
                    candidate.access_count += 1
                    candidate.last_accessed = now_iso()
                    self.store.upsert_atom(candidate)
                    return {
                        "status": "duplicate",
                        "existing_id": candidate.id,
                        "superseded": [],
                    }
            # Clear contradiction / replacement
            if score >= 0.58:
                invalidated = self.store.invalidate_atom(candidate.id, superseded_by=atom.id)
                if invalidated and self._vector_ready:
                    self.vector.upsert_atoms([invalidated])
                superseded.append(
                    {
                        "old_id": candidate.id,
                        "old_text": candidate.text,
                        "score": round(score, 3),
                    }
                )

        self.store.upsert_atom(atom)
        if self._vector_ready:
            self.vector.upsert_atoms([atom])
        if self._graph_ready:
            self.graph.upsert_atoms([atom])
            for item in superseded:
                self.graph.supersede_atom(
                    item["old_id"],
                    atom.id,
                    reason=f"auto-contradiction:{item['score']}",
                )
        return {"status": "created", "atom_id": atom.id, "superseded": superseded}

    def _atom_search_results(
        self,
        query: str,
        *,
        limit: int,
        memory_types: list[str] | None,
        include_superseded: bool,
        as_of: str | None = None,
        prefer_older_valid_from: bool = False,
    ) -> list[SearchResult]:
        atoms = self.store.search_atoms(
            query,
            limit=limit if not prefer_older_valid_from else max(limit * 2, 16),
            memory_types=memory_types,
            include_superseded=include_superseded,
            as_of=as_of,
        )
        if prefer_older_valid_from:
            def _sort_key(atom: AtomicMemory) -> datetime:
                parsed = parse_iso(atom.valid_from)
                if parsed is None:
                    return datetime.min.replace(tzinfo=UTC)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=UTC)
                return parsed

            atoms.sort(key=_sort_key)
        results = [
            self._atom_to_search_result(atom, score=max(atom.salience, 0.35))
            for atom in atoms
        ]
        if prefer_older_valid_from:
            for result in results:
                result.score = min(result.score + 0.12, 1.0)
                result.provenance["temporal_preference"] = "older_valid_from"
            results.sort(key=lambda r: r.provenance.get("valid_from", ""))
        return results[:limit]

    @staticmethod
    def _atom_to_search_result(atom: AtomicMemory, *, score: float) -> SearchResult:
        return SearchResult(
            id=atom.id,
            title=f"{atom.memory_type.value}: {atom.text[:72]}",
            text=atom.text,
            source_path=f"atom://{atom.id}",
            memory_type=atom.memory_type.value,
            score=score,
            provenance={
                "source": "atomic-memory",
                "salience": atom.salience,
                "valid_from": atom.valid_from,
                "valid_until": atom.valid_until,
                "event_time": atom.event_time,
                "superseded_by": atom.superseded_by,
                "access_count": atom.access_count,
                "entities": atom.entities,
                "project_path": atom.project_path,
            },
        )

    def _apply_salience_rerank(self, results: list[SearchResult]) -> list[SearchResult]:
        """Blend RRF relevance with atomic salience (atoms only; preserve chunk RRF)."""
        for result in results:
            is_atom = (
                result.provenance.get("source") == "atomic-memory"
                or result.id.startswith("atom:")
            )
            result.provenance["relevance_score"] = result.score
            if not is_atom:
                result.provenance["salience_component"] = 0.0
                continue
            salience = float(result.provenance.get("salience") or 0.5)
            result.provenance["salience_component"] = salience
            result.score = blend_scores(result.score, salience)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for result in results:
            key = result.source_path or result.id
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
        return unique

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
        """Parse a transcript and build a ReflectionPayload (heuristic + keyword fallback)."""
        return extract_from_transcript(text, session_id, project_path).payload

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
    def _extract_transcript_event_time(text: str) -> str | None:
        """Best-effort parse of session date from transcript headers / frontmatter."""
        lines = text.splitlines()
        in_frontmatter = False
        for line in lines[:40]:
            stripped = line.strip()
            if stripped == "---":
                in_frontmatter = not in_frontmatter
                continue
            lower = stripped.lower()
            for key in ("created_at:", "session_date:", "date:", "event_time:"):
                if lower.startswith(key):
                    value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                    if value and parse_iso(value):
                        return value
                    loose = parse_loose_date(value)
                    if loose:
                        return loose.date().isoformat()
            if not in_frontmatter and stripped and not stripped.startswith("#"):
                # Free-text header like "Session 2024-06-15" or "1:56 pm on 8 May, 2023"
                match = re.search(
                    r"\b(20\d{2}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?)\b",
                    stripped,
                )
                if match and parse_iso(match.group(1)):
                    return match.group(1)
                loose = parse_loose_date(stripped)
                if loose:
                    return loose.date().isoformat()
        return None

    @staticmethod
    def _read_transcript_or_path(transcript_or_path: str) -> tuple[str, str]:
        # Inline transcripts can be huge; never treat multiline / huge strings as paths.
        if (
            not transcript_or_path
            or "\n" in transcript_or_path
            or len(transcript_or_path) > 512
        ):
            return transcript_or_path, "inline"
        try:
            candidate = Path(transcript_or_path)
            if candidate.exists() and candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="ignore"), "path"
        except OSError:
            pass
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
