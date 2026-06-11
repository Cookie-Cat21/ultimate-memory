from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import AuditEvent


class LocalStore:
    def __init__(self, sqlite_dir: Path, audit_dir: Path, candidates_dir: Path) -> None:
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = sqlite_dir / "ultimate_memory.db"
        self.audit_dir = audit_dir
        self.candidates_dir = candidates_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memory_index (
                    id text primary key,
                    source_path text not null,
                    title text not null,
                    memory_type text not null,
                    text text not null,
                    metadata_json text not null,
                    created_at text not null
                )
                """
            )
            conn.execute(
                """
                create virtual table if not exists memory_fts using fts5(
                    id unindexed,
                    title,
                    text,
                    source_path,
                    memory_type
                )
                """
            )
            conn.execute(
                """
                create table if not exists audit_events (
                    id text primary key,
                    action text not null,
                    created_at text not null,
                    payload_json text not null,
                    source_refs_json text not null
                )
                """
            )

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        source_path: str,
        title: str,
        memory_type: str,
        text: str,
        metadata: dict,
        created_at: str,
    ) -> None:
        conn = self._connect()
        with conn:
            conn.execute(
                """
                insert into memory_index
                    (id, source_path, title, memory_type, text, metadata_json, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    source_path=excluded.source_path,
                    title=excluded.title,
                    memory_type=excluded.memory_type,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                (
                    chunk_id,
                    source_path,
                    title,
                    memory_type,
                    text,
                    json.dumps(metadata, ensure_ascii=True),
                    created_at,
                ),
            )
            conn.execute("delete from memory_fts where id = ?", (chunk_id,))
            conn.execute(
                """
                insert into memory_fts (id, title, text, source_path, memory_type)
                values (?, ?, ?, ?, ?)
                """,
                (chunk_id, title, text, source_path, memory_type),
            )

    def keyword_search(self, query: str, limit: int = 8) -> list[dict]:
        if not query.strip():
            return []
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    select id, title, text, source_path, memory_type,
                           bm25(memory_fts) as score
                    from memory_fts
                    where memory_fts match ?
                    order by score
                    limit ?
                    """,
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    """
                    select id, title, text, source_path, memory_type, 0.0 as score
                    from memory_fts
                    where title like ? or text like ?
                    limit ?
                    """,
                    (f"%{query}%", f"%{query}%", limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def write_audit(self, event: AuditEvent) -> None:
        payload_json = event.model_dump_json(indent=2)
        (self.audit_dir / f"{event.created_at[:10]}-{event.id}.json").write_text(
            payload_json,
            encoding="utf-8",
        )
        with self._connect() as conn:
            conn.execute(
                """
                insert or replace into audit_events
                    (id, action, created_at, payload_json, source_refs_json)
                values (?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.action,
                    event.created_at,
                    json.dumps(event.payload, ensure_ascii=True),
                    json.dumps(event.source_refs, ensure_ascii=True),
                ),
            )

    def recent_audit(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select id, action, created_at, payload_json, source_refs_json
                from audit_events
                order by created_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def write_candidate(self, candidate_id: str, payload: dict) -> Path:
        path = self.candidates_dir / f"{candidate_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def read_candidate(self, candidate_id: str) -> dict | None:
        path = self.candidates_dir / f"{candidate_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_candidate(self, candidate_id: str) -> None:
        path = self.candidates_dir / f"{candidate_id}.json"
        if path.exists():
            path.unlink()

    def list_candidates(self) -> list[dict]:
        candidates: list[dict] = []
        for path in sorted(self.candidates_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["_path"] = str(path)
                candidates.append(item)
            except json.JSONDecodeError:
                continue
        return candidates

    def all_indexed_sources(self) -> Iterable[str]:
        with self._connect() as conn:
            rows = conn.execute("select distinct source_path from memory_index").fetchall()
        return [row["source_path"] for row in rows]
