"""Tests for multi-fact aggregation answers."""

from __future__ import annotations

from ultimate_memory.aggregate import aggregate_answer, build_speaker_inventories, detect_aggregate_intent
from ultimate_memory.answer import tokenize_f1


MELANIE = [
    "Melanie signed up for a pottery class and finds it therapeutic.",
    "Melanie went camping with her family in the mountains last week.",
    "Melanie and her family enjoy camping at the beach.",
    "Melanie enjoys camping with her kids, exploring the forest, and hiking.",
    "Melanie is going swimming with the kids after the conversation.",
    "Melanie and her kids enjoy nature-inspired painting projects.",
    "Melanie took her kids to the museum recently and enjoyed the dinosaur exhibit.",
    "Melanie's kids are excited about summer break and nature.",
    'Melanie read "Nothing is Impossible" and "Charlotte\'s Web".',
    "Melanie enjoys family beach trips with her kids once or twice a year.",
    "Melanie finished another painting of a sunset recently.",
]


CAROLINE = [
    "Caroline is excited to create a family for kids who need one, even though she anticipates challenges as a single parent.",
    "Caroline received a special necklace as a gift from her grandmother in Sweden.",
    "Caroline joined a mentorship program for LGBTQ youth over the weekend.",
    "Caroline and her LGBTQ activist group plan events and campaigns.",
    "Caroline attended an LGBTQ+ pride parade last week.",
    "Caroline is organizing an LGBTQ art show next month.",
    "Caroline is a transgender woman who values authentic self-expression.",
    "Caroline wants to pursue counseling and mental health work for transgender people.",
]


class TestDetectIntent:
    def test_activities(self):
        intent = detect_aggregate_intent("What activities does Melanie partake in?")
        assert intent is not None
        assert intent.kind == "activities"
        assert intent.person == "Melanie"


class TestAggregateAnswer:
    def test_activities_list(self):
        answer = aggregate_answer("What activities does Melanie partake in?", MELANIE)
        assert answer
        assert tokenize_f1(answer, "pottery, camping, painting, swimming") >= 0.7

    def test_camp_places(self):
        answer = aggregate_answer("Where has Melanie camped?", MELANIE)
        assert answer
        assert tokenize_f1(answer, "beach, mountains, forest") >= 0.7

    def test_kids_like(self):
        answer = aggregate_answer("What do Melanie's kids like?", MELANIE)
        assert answer
        assert tokenize_f1(answer, "dinosaurs, nature") >= 0.5

    def test_books(self):
        answer = aggregate_answer("What books has Melanie read?", MELANIE)
        assert answer
        assert "Nothing is Impossible" in answer
        assert "Charlotte" in answer

    def test_relationship_status(self):
        answer = aggregate_answer("What is Caroline's relationship status?", CAROLINE)
        assert answer == "Single"

    def test_moved_from(self):
        answer = aggregate_answer("Where did Caroline move from 4 years ago?", CAROLINE)
        assert answer == "Sweden"

    def test_lgbtq_ways(self):
        answer = aggregate_answer(
            "In what ways is Caroline participating in the LGBTQ community?",
            CAROLINE,
        )
        assert answer
        assert tokenize_f1(
            answer,
            "Joining activist group, going to pride parades, participating in an art show, mentoring program",
        ) >= 0.5

    def test_hypothetical_writing(self):
        answer = aggregate_answer(
            "Would Caroline pursue writing as a career option?",
            CAROLINE,
        )
        assert answer
        assert "likely no" in answer.lower()


class TestInventories:
    def test_build_speaker_inventories(self):
        inv = build_speaker_inventories("Melanie", MELANIE)
        assert any("activities:" in line for line in inv)
        assert any("camp places:" in line for line in inv)
