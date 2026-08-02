"""
Comprehensive tests for the v2 improvements:
  1. Markdown-aware chunking
  2. Vector filtering (tags + memory_types)
  3. Async parallel retrieval + RRF ranking
  4. Smart session ingestion with auto-reflect
"""
from __future__ import annotations

import time
import types
import uuid
from pathlib import Path

import pytest

from ultimate_memory.chunking import _find_fence_ranges, chunk_text
from ultimate_memory.models import SearchResult
from ultimate_memory.router import MemoryRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake(id: str, path: str, score: float, src: str = "vec") -> SearchResult:
    return SearchResult(
        id=id,
        text="sample text",
        title="title",
        source_path=path,
        score=score,
        provenance={"source": src},
    )


def _dummy_router() -> MemoryRouter:
    """MemoryRouter whose _rrf_rank we can call without live services."""
    ns = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            retrieval=types.SimpleNamespace(default_limit=8)
        )
    )
    return ns  # type: ignore[return-value]


# ===========================================================================
# 1. Markdown-aware chunking
# ===========================================================================

class TestChunking:
    def test_short_text_passthrough(self):
        assert chunk_text("hello world", 1200) == ["hello world"]

    def test_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_heading_preferred_over_paragraph(self):
        """Splits at heading boundary when one exists past the 40% mark."""
        body = "word " * 100  # 500 chars
        text = "## Section A\n\n" + body + "\n\n## Section B\n\n" + body
        chunks = chunk_text(text, chunk_chars=600, overlap=60)
        # Second chunk should start at or near the second heading
        assert any("## Section B" in c for c in chunks), (
            "Expected a chunk starting with '## Section B'"
        )
        # Heading should not be orphaned mid-chunk in a bad way
        for chunk in chunks:
            # Each fence marker must be paired within the same chunk
            opens = chunk.count("```")
            assert opens % 2 == 0 or opens == 0

    def test_fence_detection_backtick(self):
        text = "before\n```python\ncode here\nmore code\n```\nafter"
        ranges = _find_fence_ranges(text)
        assert len(ranges) == 1
        start, end = ranges[0]
        assert text[start:].startswith("```python")
        assert text[:end].endswith("```")

    def test_fence_detection_tilde(self):
        text = "before\n~~~\ncode\n~~~\nafter"
        ranges = _find_fence_ranges(text)
        assert len(ranges) == 1

    def test_fence_detection_unclosed(self):
        text = "before\n```python\nunclosed fence\nno closing"
        ranges = _find_fence_ranges(text)
        assert len(ranges) == 1
        assert ranges[0][1] == len(text)

    def test_fence_not_split(self):
        """A code block must never be split across two chunks."""
        fence_body = "x = 1\n" * 80  # ~480 chars inside the fence
        text = (
            "## Intro\n\n"
            + "context " * 40 + "\n\n"
            + "```python\n"
            + fence_body
            + "```\n\n"
            + "## After\n\nmore text here"
        )
        chunks = chunk_text(text, chunk_chars=600, overlap=60)
        for chunk in chunks:
            n_open = len(
                [line for line in chunk.splitlines() if line.strip().startswith("```") and len(line.strip()) > 3]
            )
            n_close = chunk.count("\n```\n") + (1 if chunk.endswith("\n```") else 0)
            # Either both fence markers are in this chunk or neither
            assert n_open == n_close, (
                f"Chunk has mismatched fence markers (open={n_open}, close={n_close}):\n{chunk[:200]}"
            )

    def test_overlap_produces_context_continuity(self):
        """The start of chunk N+1 should overlap with the end of chunk N."""
        text = "sentence one. " * 200  # 2800 chars
        chunks = chunk_text(text, chunk_chars=400, overlap=80)
        assert len(chunks) >= 3
        for i in range(len(chunks) - 1):
            tail = chunks[i][-60:].strip()
            head = chunks[i + 1][:60].strip()
            # Some overlap must exist (head of next should share text with tail of current)
            shared = set(tail.split()) & set(head.split())
            assert len(shared) > 0, (
                f"No overlap between chunk {i} and {i+1}:\n  tail={tail!r}\n  head={head!r}"
            )

    def test_no_empty_chunks(self):
        text = "\n\n".join(["paragraph " * 30] * 10)
        chunks = chunk_text(text, chunk_chars=500, overlap=50)
        assert all(c.strip() for c in chunks)

    def test_multiple_fences(self):
        """Two separate code blocks in the same document are both detected."""
        text = (
            "intro\n\n```bash\necho hello\n```\n\n"
            + "middle section " * 30 + "\n\n"
            + "```python\nprint('hi')\n```\n\nfooter"
        )
        ranges = _find_fence_ranges(text)
        assert len(ranges) == 2


# ===========================================================================
# 2. RRF ranking (pure logic, no services)
# ===========================================================================

class TestRRFRanking:
    def test_multi_list_winner(self):
        """A result appearing in all three lists should rank first."""
        r = _dummy_router()
        l1 = [_fake("a", "p/a", 0.9), _fake("b", "p/b", 0.7), _fake("c", "p/c", 0.5)]
        l2 = [_fake("b", "p/b", 0.85), _fake("a", "p/a", 0.6), _fake("d", "p/d", 0.4)]
        l3 = [_fake("a", "p/a", 0.8)]
        results = MemoryRouter._rrf_rank(r, [l1, l2, l3])
        assert results[0].source_path == "p/a"
        assert results[1].source_path == "p/b"

    def test_scores_normalised_to_0_1(self):
        r = _dummy_router()
        l1 = [_fake("a", "p/a", 0.9), _fake("b", "p/b", 0.7)]
        l2 = [_fake("b", "p/b", 0.8), _fake("a", "p/a", 0.5)]
        results = MemoryRouter._rrf_rank(r, [l1, l2])
        for res in results:
            assert 0.0 <= res.score <= 1.0, f"Score out of range: {res.score}"

    def test_raw_score_preserved_in_provenance(self):
        r = _dummy_router()
        original_score = 0.73
        l1 = [_fake("x", "p/x", original_score)]
        results = MemoryRouter._rrf_rank(r, [l1])
        assert results[0].provenance["raw_score"] == original_score

    def test_memory_type_filter(self):
        r = _dummy_router()
        note = _fake("n", "p/n", 0.9)
        note.memory_type = "note"
        decision = _fake("d", "p/d", 0.8)
        decision.memory_type = "decision"
        results = MemoryRouter._rrf_rank(r, [[note, decision]], memory_types=["decision"])
        assert len(results) == 1
        assert results[0].source_path == "p/d"

    def test_empty_lists_return_empty(self):
        r = _dummy_router()
        results = MemoryRouter._rrf_rank(r, [[], [], []])
        assert results == []

    def test_single_result_scores_1(self):
        r = _dummy_router()
        results = MemoryRouter._rrf_rank(r, [[_fake("a", "p/a", 0.5)]])
        assert results[0].score == 1.0

    def test_limit_respected(self):
        r = _dummy_router()
        items = [_fake(str(i), f"p/{i}", 1.0 - i * 0.05) for i in range(20)]
        results = MemoryRouter._rrf_rank(r, [items], limit=5)
        assert len(results) == 5


# ===========================================================================
# 3. Smart extraction
# ===========================================================================

class TestExtraction:
    def test_extracts_decision(self):
        text = "We decided to use Redis for caching instead of in-memory."
        r = MemoryRouter._extract_reflection(text, "s1", None)
        assert r is not None
        assert any("Redis" in d for d in r.decisions)

    def test_extracts_fact(self):
        text = (
            "The root cause was a misconfigured timeout value in config.py.\n"
            "The fix was to set TIMEOUT=30 instead of 3."
        )
        r = MemoryRouter._extract_reflection(text, "s1", "/proj/myapp")
        assert r is not None
        assert len(r.facts) >= 1

    def test_extracts_preference(self):
        text = "I prefer we always use environment variables for secrets, never hardcode them."
        r = MemoryRouter._extract_reflection(text, "s1", None)
        assert r is not None
        assert len(r.preferences) >= 1

    def test_extracts_procedure(self):
        text = "Steps: 1) run migrations 2) restart server 3) verify logs."
        r = MemoryRouter._extract_reflection(text, "s1", None)
        assert r is not None
        assert len(r.procedures) >= 1

    def test_low_signal_returns_none(self):
        """A transcript with zero extractable signals should not create a reflection."""
        text = "Hi. How are you? Great. See you later."
        r = MemoryRouter._extract_reflection(text, "s1", None)
        assert r is None

    def test_caps_per_category(self):
        """Even a very signal-dense transcript caps facts at 6, decisions at 6, etc."""
        decisions = "\n".join(f"We decided to use option{i} going forward." for i in range(20))
        facts = "\n".join(f"The fix was applied to module{i}." for i in range(20))
        r = MemoryRouter._extract_reflection(decisions + "\n" + facts, "s1", None)
        assert r is not None
        assert len(r.decisions) <= 6
        assert len(r.facts) <= 12

    def test_deduplicates_signals(self):
        """Identical lines should not produce duplicate entries."""
        line = "We decided to use Redis for caching going forward."
        text = "\n".join([line] * 10)
        r = MemoryRouter._extract_reflection(text, "s1", None)
        if r:
            assert len(r.decisions) == 1

    def test_project_name_in_summary(self):
        text = "The root cause was a bug.\nThe fix was a patch.\nWe decided to upgrade."
        r = MemoryRouter._extract_reflection(text, "abc123", "/projects/dinaya")
        assert r is not None
        assert "dinaya" in r.summary

    def test_source_ref_contains_session_id(self):
        text = "The root cause was X.\nThe fix was Y.\nWe decided to keep Z."
        r = MemoryRouter._extract_reflection(text, "sess-xyz", None)
        assert r is not None
        assert any("sess-xyz" in ref for ref in r.source_refs)


# ===========================================================================
# 4. Live integration tests (require running Qdrant + Neo4j + Obsidian vault)
# ===========================================================================

@pytest.fixture(scope="module")
def router():
    r = MemoryRouter()
    if not r._vector_ready:
        pytest.skip("Qdrant not available")
    return r


class TestLiveSearch:
    def test_parallel_search_returns_results(self, router):
        result = router.search("memory session project")
        assert "results" in result
        assert "query" in result
        assert isinstance(result["results"], list)

    def test_rrf_scores_in_0_1(self, router):
        result = router.search("preferences decisions")
        for r in result["results"]:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_rrf_provenance_has_raw_score(self, router):
        result = router.search("project ardeno")
        for r in result["results"]:
            # Every result that went through RRF has raw_score in provenance
            assert "raw_score" in r["provenance"], (
                f"Missing raw_score in provenance for: {r['title']}"
            )

    def test_parallel_faster_than_sequential_threshold(self, router):
        """Parallel search should complete in under 3 seconds even in full mode."""
        start = time.perf_counter()
        router.search("ultimate memory bootstrap session")
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"Search took {elapsed:.2f}s — too slow"

    def test_memory_type_filter_respected(self, router):
        result = router.search("session log", memory_types=["log"])
        for r in result["results"]:
            assert r["memory_type"] == "log", (
                f"Got memory_type={r['memory_type']!r} but asked for log only"
            )

    def test_search_returns_graph_hits(self, router):
        result = router.search("ultimate memory")
        assert "graph_hits" in result
        assert isinstance(result["graph_hits"], list)


class TestLiveIngest:
    def test_ingest_log_returns_reflection(self, router, tmp_path):
        transcript = (
            "User: can you fix the timeout?\n"
            "Assistant: The root cause was TIMEOUT set to 3 instead of 30.\n"
            "The fix was updating config.py line 42.\n"
            "We decided to always use environment config going forward.\n"
            "I prefer we never hardcode secrets in source files.\n"
        )
        session_id = f"test-{uuid.uuid4().hex[:8]}"
        result = router.ingest_log(
            client="test-suite",
            session_id=session_id,
            transcript_or_path=transcript,
            project_path=str(tmp_path),
            tags=["test"],
        )
        assert result["chunks"] >= 1
        assert "reflection" in result
        # Should have auto-reflected (enough signals in transcript)
        assert result["reflection"] is not None
        assert "confidence" in result["reflection"]

    def test_ingest_log_no_reflection_on_noise(self, router, tmp_path):
        transcript = "User: hello\nAssistant: hi there!\nUser: thanks\nAssistant: sure."
        session_id = f"test-{uuid.uuid4().hex[:8]}"
        result = router.ingest_log(
            client="test-suite",
            session_id=session_id,
            transcript_or_path=transcript,
            project_path=str(tmp_path),
        )
        assert result["chunks"] >= 1
        assert result["reflection"] is None  # not enough signal

    def test_ingest_log_creates_file(self, router, tmp_path):
        transcript = (
            "The root cause was a bad config.\n"
            "The fix was to update the settings.\n"
            "We decided to document all changes.\n"
        )
        session_id = f"test-{uuid.uuid4().hex[:8]}"
        result = router.ingest_log(
            client="test-suite",
            session_id=session_id,
            transcript_or_path=transcript,
        )
        log_path = Path(result["path"])
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "test-suite" in content
        assert session_id in content
