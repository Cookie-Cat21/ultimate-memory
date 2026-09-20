from ultimate_memory.models import SearchResult
from ultimate_memory.planner import plan_query
from ultimate_memory.ranking import filter_entity_scoped_results, rerank_candidates


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


def test_multi_entity_filter_balances_named_people():
    results = [
        {"id": "a1", "text": "Jean visited Paris.", "provenance": {}},
        {"id": "a2", "text": "Jean visited Rome.", "provenance": {}},
        {"id": "a3", "text": "Jean likes museums.", "provenance": {}},
        {"id": "b1", "text": "John visited Rome.", "provenance": {}},
    ]
    balanced = filter_entity_scoped_results(results, ["Jean", "John"])
    assert balanced[0]["id"] == "a1"
    assert balanced[1]["id"] == "b1"


def test_speaker_entity_takes_priority_over_capitalized_topic():
    results = [
        {
            "id": "caroline",
            "text": "Caroline attended a pride event.",
            "provenance": {"speaker": "Caroline"},
        },
        {
            "id": "melanie",
            "text": "Melanie discussed LGBTQ community events.",
            "provenance": {"speaker": "Melanie"},
        },
    ]
    scoped = filter_entity_scoped_results(results, ["LGBTQ", "Caroline"], min_matches=1)
    assert [item["id"] for item in scoped] == ["caroline"]


def test_question_speaker_only_pair_is_weak_evidence():
    results = [
        {
            "id": "pair",
            "text": "Melanie: What books have you read?\nCaroline: I read Dune.",
            "provenance": {
                "pair": True,
                "question_speaker": "Melanie",
                "answer_speaker": "Caroline",
            },
        },
        {
            "id": "direct",
            "text": "Melanie: I read The Hobbit last year.",
            "provenance": {"speaker": "Melanie", "direct_turn": True},
        },
    ]
    scoped = filter_entity_scoped_results(results, ["Melanie"], min_matches=1)
    assert scoped[0]["id"] == "direct"
    assert {item["id"] for item in scoped} == {"direct", "pair"}


def test_pair_answer_speaker_is_strong_evidence():
    results = [
        {
            "id": "pair",
            "text": "Caroline: What books have you read?\nMelanie: I read The Hobbit.",
            "provenance": {
                "pair": True,
                "question_speaker": "Caroline",
                "answer_speaker": "Melanie",
            },
        },
        {
            "id": "noise",
            "text": "Caroline mentioned Melanie while discussing art.",
            "provenance": {},
        },
    ]
    scoped = filter_entity_scoped_results(results, ["Melanie"], min_matches=1)
    assert scoped[0]["id"] == "pair"


def test_nested_vector_payload_preserves_answer_attribution():
    results = [
        {
            "id": "vector-pair",
            "text": "Caroline: What do you do to relax?\nMelanie: I go running.",
            "provenance": {
                "source": "qdrant",
                "payload": {
                    "pair": True,
                    "question_speaker": "Caroline",
                    "answer_speaker": "Melanie",
                },
            },
        }
    ]
    scoped = filter_entity_scoped_results(results, ["Melanie"], min_matches=1)
    assert [item["id"] for item in scoped] == ["vector-pair"]
