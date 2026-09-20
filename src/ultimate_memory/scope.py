"""Hierarchical scope helpers for user/org/project/repo/branch/session memory."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryScope(BaseModel):
    user: str | None = None
    organization: str | None = None
    project: str | None = None
    repository: str | None = None
    branch: str | None = None
    session: str | None = None
    tags: list[str] = Field(default_factory=list)

    def specificity(self) -> int:
        return sum(
            value is not None
            for value in (
                self.user,
                self.organization,
                self.project,
                self.repository,
                self.branch,
                self.session,
            )
        )

    def as_metadata(self) -> dict:
        return {"scope": self.model_dump(exclude_none=True)}

    def matches(self, requested: "MemoryScope") -> bool:
        """A stored scope matches when it does not contradict requested ancestors."""
        for field in ("user", "organization", "project", "repository", "branch", "session"):
            stored = getattr(self, field)
            wanted = getattr(requested, field)
            if stored is not None and wanted is not None and stored != wanted:
                return False
        return True


def scope_boost(stored: MemoryScope, requested: MemoryScope) -> float:
    """Prefer the most specific compatible scope without excluding global memory."""
    if not stored.matches(requested):
        return -1.0
    shared = 0
    for field in ("user", "organization", "project", "repository", "branch", "session"):
        value = getattr(stored, field)
        if value is not None and value == getattr(requested, field):
            shared += 1
    return min(0.24, shared * 0.04)
