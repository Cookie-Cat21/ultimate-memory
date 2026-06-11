"""
Called by the Claude Code SessionEnd hook via ingest-claude-session.ps1.
Reads JSON from stdin: {session_id, transcript_path, project_path}
and ingests the session transcript into Ultimate Memory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultimate_memory.router import MemoryRouter


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return

    session_id = data.get("session_id") or "unknown"
    transcript_path = data.get("transcript_path") or ""
    project_path = data.get("project_path")

    if not transcript_path:
        return

    try:
        router = MemoryRouter()
        result = router.ingest_log(
            client="claude-code",
            session_id=session_id,
            transcript_or_path=transcript_path,
            project_path=project_path,
        )
        print(json.dumps(result))
    except Exception as exc:
        print(f"ultimate-memory ingest error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
