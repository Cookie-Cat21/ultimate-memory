from ultimate_memory.models import SearchResult
from ultimate_memory.planner import plan_query
from ultimate_memory.ranking import rerank_candidates


def result(text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        id=text,
        text=text,
        title=text,
        source_path=text,
        memory_type="fact",
        score=score,
        provenance={},
    )


def test_matching_duration_quantity_beats_conflicting_duration():
    query = "Where did Caroline move from 4 years ago?"
    ranked = rerank_candidates(
        query,
        [
            result("Caroline moved from Sweden four years ago."),
            result("Caroline started guitar five years ago."),
        ],
        plan_query(query),
    )
    assert "Sweden" in ranked[0].text


def test_recent_query_prefers_newer_event_time():
    query = "What did Melanie paint recently?"
    plan = plan_query(query)
    old = result("Melanie painted horses.")
    old.provenance["event_time"] = "2023-01-01T00:00:00+00:00"
    new = result("Melanie painted a sunset.")
    new.provenance["event_time"] = "2023-10-01T00:00:00+00:00"
    ranked = rerank_candidates(query, [old, new], plan)
    assert "sunset" in ranked[0].text
