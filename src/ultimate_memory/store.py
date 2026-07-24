from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from .atoms import atom_valid_at, compute_salience, content_hash, now_iso
from .models import AtomicMemory, AuditEvent, MemoryType

_FTS_QUERY_STOP = frozenset(
    {
        "where",
        "what",
        "when",
        "who",
        "how",
        "why",
        "did",
        "does",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "used",
        "use",
        "live",
        "lives",
        "lived",
        "was",
        "were",
        "are",
        "have",
        "has",
        "had",
    }
)


def _atom_fts_query_variants(query: str) -> list[str]:
    """Build FTS5-safe query variants for natural-language atom search."""
    stripped = query.strip()
    variants: list[str] = []
    if stripped:
        variants.append(stripped)
    terms: list[str] = []
    for match in re.finditer(r"[a-zA-Z]{3,}", stripped):
        token = match.group(0).lower()
        if token not in _FTS_QUERY_STOP:
            terms.append(token)
    if terms:
        or_query = " OR ".join(dict.fromkeys(terms))
        if or_query not in variants:
            variants.append(or_query)
    return variants


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
            conn.execute(
                """
                create table if not exists memory_atoms (
                    id text primary key,
                    text text not null,
                    memory_type text not null,
                    project_path text,
                    entities_json text not null,
                    source_refs_json text not null,
                    created_at text not null,
                    valid_from text not null,
                    valid_until text,
                    superseded_by text,
                    salience real not null,
                    access_count integer not null,
                    last_accessed text,
                    importance real not null,
                    content_hash text not null,
                    metadata_json text not null
                )
                """
            )
            conn.execute(
                """
                create index if not exists idx_atoms_type_active
                on memory_atoms(memory_type, valid_until)
                """
            )
            conn.execute(
                """
                create index if not exists idx_atoms_hash
                on memory_atoms(content_hash)
                """
            )
            conn.execute(
                """
                create virtual table if not exists memory_atoms_fts using fts5(
                    id unindexed,
                    text,
                    memory_type,
                    project_path
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

    def upsert_atom(self, atom: AtomicMemory) -> AtomicMemory:
        if not atom.content_hash:
            atom.content_hash = content_hash(atom.text)
        if atom.event_time:
            atom.metadata["event_time"] = atom.event_time
        elif atom.metadata.get("event_time"):
            atom.event_time = atom.metadata["event_time"]
        atom.salience = compute_salience(
            atom.memory_type.value,
            created_at=atom.created_at,
            last_accessed=atom.last_accessed,
            access_count=atom.access_count,
            valid_until=atom.valid_until,
            importance=atom.importance,
        )
        conn = self._connect()
        with conn:
            conn.execute(
                """
                insert into memory_atoms (
                    id, text, memory_type, project_path, entities_json, source_refs_json,
                    created_at, valid_from, valid_until, superseded_by, salience,
                    access_count, last_accessed, importance, content_hash, metadata_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    text=excluded.text,
                    memory_type=excluded.memory_type,
                    project_path=excluded.project_path,
                    entities_json=excluded.entities_json,
                    source_refs_json=excluded.source_refs_json,
                    created_at=excluded.created_at,
                    valid_from=excluded.valid_from,
                    valid_until=excluded.valid_until,
                    superseded_by=excluded.superseded_by,
                    salience=excluded.salience,
                    access_count=excluded.access_count,
                    last_accessed=excluded.last_accessed,
                    importance=excluded.importance,
                    content_hash=excluded.content_hash,
                    metadata_json=excluded.metadata_json
                """,
                (
                    atom.id,
                    atom.text,
                    atom.memory_type.value,
                    atom.project_path,
                    json.dumps(atom.entities, ensure_ascii=True),
                    json.dumps(atom.source_refs, ensure_ascii=True),
                    atom.created_at,
                    atom.valid_from,
                    atom.valid_until,
                    atom.superseded_by,
                    atom.salience,
                    atom.access_count,
                    atom.last_accessed,
                    atom.importance,
                    atom.content_hash,
                    json.dumps(atom.metadata, ensure_ascii=True),
                ),
            )
            conn.execute("delete from memory_atoms_fts where id = ?", (atom.id,))
            conn.execute(
                """
                insert into memory_atoms_fts (id, text, memory_type, project_path)
                values (?, ?, ?, ?)
                """,
                (atom.id, atom.text, atom.memory_type.value, atom.project_path or ""),
            )
        return atom

    def get_atom(self, atom_id: str) -> AtomicMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from memory_atoms where id = ?",
                (atom_id,),
            ).fetchone()
        return self._row_to_atom(row) if row else None

    def list_active_atoms(
        self,
        *,
        memory_types: list[str] | None = None,
        project_path: str | None = None,
        limit: int = 100,
        as_of: str | None = None,
    ) -> list[AtomicMemory]:
        if as_of:
            clauses = ["valid_from <= ?", "(valid_until is null or valid_until > ?)"]
            params: list[object] = [as_of, as_of]
        else:
            clauses = ["valid_until is null", "superseded_by is null"]
            params = []
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"memory_type in ({placeholders})")
            params.extend(memory_types)
        if project_path:
            clauses.append("(project_path = ? or project_path is null or project_path = '')")
            params.append(project_path)
        params.append(limit)
        sql = f"""
            select * from memory_atoms
            where {' and '.join(clauses)}
            order by salience desc, created_at desc
            limit ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def search_atoms(
        self,
        query: str,
        *,
        limit: int = 8,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        as_of: str | None = None,
    ) -> list[AtomicMemory]:
        if not query.strip():
            return []
        atoms: list[AtomicMemory] = []
        rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            for fts_query in _atom_fts_query_variants(query):
                try:
                    candidate_rows = conn.execute(
                        """
                        select a.*
                        from memory_atoms_fts f
                        join memory_atoms a on a.id = f.id
                        where memory_atoms_fts match ?
                        order by bm25(memory_atoms_fts)
                        limit ?
                        """,
                        (fts_query, max(limit * 4, 16)),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if candidate_rows:
                    rows = candidate_rows
                    break
            if not rows:
                like_terms = [
                    t
                    for t in re.findall(r"[a-zA-Z]{3,}", query)
                    if t.lower() not in _FTS_QUERY_STOP
                ]
                if like_terms:
                    pattern = f"%{like_terms[0]}%"
                    rows = conn.execute(
                        """
                        select * from memory_atoms
                        where text like ?
                        order by salience desc
                        limit ?
                        """,
                        (pattern, max(limit * 4, 16)),
                    ).fetchall()
        allowed = set(memory_types or [])
        for row in rows:
            atom = self._row_to_atom(row)
            if as_of:
                if not atom_valid_at(atom, as_of):
                    continue
            elif not include_superseded and not atom.is_active:
                continue
            if allowed and atom.memory_type.value not in allowed:
                continue
            atoms.append(atom)
            if len(atoms) >= limit:
                break
        return atoms

    def find_duplicate_atom(self, atom: AtomicMemory) -> AtomicMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from memory_atoms
                where content_hash = ?
                  and memory_type = ?
                  and valid_until is null
                  and superseded_by is null
                  and id != ?
                limit 1
                """,
                (atom.content_hash or content_hash(atom.text), atom.memory_type.value, atom.id),
            ).fetchone()
        return self._row_to_atom(row) if row else None

    def find_contradiction_candidates(
        self,
        atom: AtomicMemory,
        *,
        limit: int = 12,
    ) -> list[AtomicMemory]:
        """Return active same-type atoms that may contradict *atom*."""
        # Prefer same project, then global actives of that type.
        same_project = self.list_active_atoms(
            memory_types=[atom.memory_type.value],
            project_path=atom.project_path,
            limit=limit,
        )
        if len(same_project) >= limit:
            return [a for a in same_project if a.id != atom.id]
        extras = self.list_active_atoms(memory_types=[atom.memory_type.value], limit=limit * 2)
        seen = {a.id for a in same_project}
        merged = [a for a in same_project if a.id != atom.id]
        for candidate in extras:
            if candidate.id == atom.id or candidate.id in seen:
                continue
            merged.append(candidate)
            if len(merged) >= limit:
                break
        return merged

    def invalidate_atom(
        self,
        atom_id: str,
        *,
        superseded_by: str,
        valid_until: str | None = None,
    ) -> AtomicMemory | None:
        atom = self.get_atom(atom_id)
        if not atom:
            return None
        atom.valid_until = valid_until or now_iso()
        atom.superseded_by = superseded_by
        atom.salience = 0.0
        return self.upsert_atom(atom)

    def touch_atoms(self, atom_ids: Iterable[str]) -> None:
        stamp = now_iso()
        ids = list(dict.fromkeys(atom_ids))
        if not ids:
            return
        conn = self._connect()
        with conn:
            for atom_id in ids:
                row = conn.execute(
                    "select * from memory_atoms where id = ?",
                    (atom_id,),
                ).fetchone()
                if not row:
                    continue
                atom = self._row_to_atom(row)
                if not atom.is_active:
                    continue
                atom.access_count += 1
                atom.last_accessed = stamp
                atom.salience = compute_salience(
                    atom.memory_type.value,
                    created_at=atom.created_at,
                    last_accessed=atom.last_accessed,
                    access_count=atom.access_count,
                    valid_until=atom.valid_until,
                    importance=atom.importance,
                )
                conn.execute(
                    """
                    update memory_atoms
                    set access_count = ?, last_accessed = ?, salience = ?
                    where id = ?
                    """,
                    (atom.access_count, atom.last_accessed, atom.salience, atom.id),
                )

    def refresh_salience(self, limit: int = 500) -> int:
        """Recompute salience for active atoms (decay over time)."""
        atoms = self.list_active_atoms(limit=limit)
        count = 0
        for atom in atoms:
            new_salience = compute_salience(
                atom.memory_type.value,
                created_at=atom.created_at,
                last_accessed=atom.last_accessed,
                access_count=atom.access_count,
                valid_until=atom.valid_until,
                importance=atom.importance,
            )
            if abs(new_salience - atom.salience) > 1e-6:
                atom.salience = new_salience
                self.upsert_atom(atom)
                count += 1
        return count

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

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> AtomicMemory:
        metadata = json.loads(row["metadata_json"] or "{}")
        event_time = metadata.get("event_time")
        return AtomicMemory(
            id=row["id"],
            text=row["text"],
            memory_type=MemoryType(row["memory_type"]),
            project_path=row["project_path"],
            entities=json.loads(row["entities_json"] or "[]"),
            source_refs=json.loads(row["source_refs_json"] or "[]"),
            created_at=row["created_at"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            event_time=event_time,
            superseded_by=row["superseded_by"],
            salience=float(row["salience"]),
            access_count=int(row["access_count"]),
            last_accessed=row["last_accessed"],
            importance=float(row["importance"]),
            content_hash=row["content_hash"],
            metadata=metadata,
        )
