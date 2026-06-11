from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    NOTE = "note"
    LOG = "log"
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    PROCEDURE = "procedure"
    AUDIT = "audit"
    CANDIDATE = "candidate"


class SourceRef(BaseModel):
    id: str
    path: str | None = None
    title: str | None = None
    url: str | None = None
    memory_type: str | None = None
    score: float | None = None


class MemoryChunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str
    source_path: str
    title: str
    memory_type: MemoryType = MemoryType.NOTE
    project_path: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    id: str
    text: str
    title: str
    source_path: str | None = None
    memory_type: str = "note"
    score: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)


class ReflectionPayload(BaseModel):
    source_refs: list[str] = Field(default_factory=list)
    summary: str
    facts: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    action: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    payload: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)


def safe_slug(value: str, fallback: str = "memory") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:96] or fallback


def path_to_uri(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")

