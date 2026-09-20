"""Answer-stage temporal context rendering.

Retrieval scores remain unchanged. When the query is temporal, memories that
carry a trusted event timestamp expose that timestamp to the answer reader.
"""
from __future__ import annotations

from typing import Any


def temporal_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for item in contexts:
        clone = dict(item)
        provenance = clone.get("provenance")
        if not isinstance(provenance, dict):
            rendered.append(clone)
            continue
        event_time = str(provenance.get("event_time") or "").strip()
        text = str(clone.get("text") or "").strip()
        if event_time and text:
            clone["text"] = f"[Memory date: {event_time}] {text}"
            rendered_provenance = dict(provenance)
            rendered_provenance["temporal_context_rendered"] = True
            clone["provenance"] = rendered_provenance
        rendered.append(clone)
    return rendered
