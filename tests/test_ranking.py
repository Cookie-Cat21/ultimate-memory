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
