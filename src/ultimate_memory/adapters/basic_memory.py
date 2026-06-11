from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..chunking import chunk_text
from ..config import BasicMemoryConfig, RetrievalConfig
from ..models import MemoryChunk, MemoryType, SearchResult, safe_slug


class BasicMemoryAdapter:
    def __init__(
        self,
        vault_path: Path,
        config: BasicMemoryConfig,
        retrieval: RetrievalConfig,
    ) -> None:
        self.vault_path = vault_path
        self.config = config
        self.retrieval = retrieval

    def is_available(self) -> bool:
        return self.vault_path.exists()

    def iter_markdown_files(self) -> list[Path]:
        if not self.vault_path.exists():
            return []
        return [
            path
            for path in self.vault_path.rglob("*.md")
            if ".obsidian" not in path.parts and path.is_file()
        ]

    def note_chunks(self) -> list[MemoryChunk]:
        chunks: list[MemoryChunk] = []
        for path in self.iter_markdown_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            rel = path.relative_to(self.vault_path)
            title = path.stem
            for idx, chunk in enumerate(
                chunk_text(
                    text,
                    self.retrieval.chunk_chars,
                    self.retrieval.chunk_overlap,
                )
            ):
                chunks.append(
                    MemoryChunk(
                        id=f"note:{safe_slug(str(rel))}:{idx}",
                        text=chunk,
                        source_path=str(path),
                        title=title,
                        memory_type=MemoryType.NOTE,
                        metadata={"relative_path": str(rel), "chunk_index": idx},
                    )
                )
        return chunks

    def search_cli(self, query: str, limit: int = 8) -> list[SearchResult]:
        command = [
            self.config.cli,
            "tool",
            "search-notes",
            query,
            "--page-size",
            str(limit),
            "--project",
            self.config.project,
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        if completed.returncode != 0:
            return []
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {"results": []}
        results: list[SearchResult] = []
        for idx, item in enumerate(payload.get("results", [])[:limit]):
            text = item.get("matched_chunk") or item.get("content") or ""
            file_path = item.get("file_path")
            source_path = str(self.vault_path / file_path) if file_path else None
            results.append(
                SearchResult(
                    id=f"basic-memory-cli-{idx}",
                    title=item.get("title") or "Basic Memory result",
                    text=text,
                    source_path=source_path,
                    memory_type=MemoryType.NOTE.value,
                    score=float(item.get("score") or 0.65),
                    provenance={
                        "source": "basic-memory-cli",
                        "permalink": item.get("permalink"),
                        "entity": item.get("entity"),
                    },
                )
            )
        return results

    def direct_search(self, query: str, limit: int = 8) -> list[SearchResult]:
        needle = query.lower().strip()
        if not needle:
            return []
        scored: list[tuple[float, Path, str]] = []
        terms = [term for term in needle.split() if len(term) > 1]
        for path in self.iter_markdown_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            haystack = text.lower()
            raw_score = sum(haystack.count(term) for term in terms)
            score = 0.35 + min(raw_score * 0.03, 0.3)
            if needle in haystack:
                score += 0.25
            if score:
                scored.append((float(score), path, text[:1200]))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                id=f"basic-memory-direct-{idx}",
                title=path.stem,
                text=snippet,
                source_path=str(path),
                memory_type=MemoryType.NOTE.value,
                score=score,
                provenance={"source": "basic-memory-direct"},
            )
            for idx, (score, path, snippet) in enumerate(scored[:limit])
        ]

    def search(self, query: str, limit: int = 8, include_cli: bool = False) -> list[SearchResult]:
        cli_results = self.search_cli(query, limit) if include_cli else []
        direct_results = self.direct_search(query, limit)
        combined = cli_results + direct_results
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in combined:
            key = result.source_path or result.text[:100]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

    def write_note(self, title: str, folder: str, content: str, tags: list[str] | None = None) -> bool:
        command = [
            self.config.cli,
            "tool",
            "write-note",
            "--title",
            title,
            "--folder",
            folder,
            "--project",
            self.config.project,
            "--local",
        ]
        if tags:
            command.extend(["--tags", ",".join(tags)])
        try:
            completed = subprocess.run(
                command,
                input=content,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return self._write_markdown_direct(title, folder, content)
        if completed.returncode == 0:
            return True
        return self._write_markdown_direct(title, folder, content)

    def _write_markdown_direct(self, title: str, folder: str, content: str) -> bool:
        target_dir = self.vault_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{title}.md"
        path.write_text(content, encoding="utf-8")
        return True
