"""Tests for heuristic session→memory extraction (LoCoMo-style dialogue)."""

from __future__ import annotations

from ultimate_memory.atoms import contradiction_score, role_conflict_boost
from ultimate_memory.extraction import extract_from_transcript, parse_dialogue_turns
from ultimate_memory.router import MemoryRouter


LOCOMO_TRANSCRIPT = """
Caroline: I work as a nurse at Seattle Children's Hospital.
Melanie: That's wonderful! I have two kids and a golden retriever named Max.
Caroline: I moved to Seattle last year from Boston.
Melanie: I always prefer hiking in the mountains on weekends.
Caroline: I decided to adopt a rescue dog next month.
Melanie: How to train a puppy: first crate-train, then work on recall daily.
Caroline: I'm allergic to shellfish and peanuts.
Melanie said she graduated from Oregon State in 2018.
"""


class TestHeuristicExtraction:
    def test_extracts_job_fact_with_speaker(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        assert result.payload is not None
        joined = " ".join(result.payload.facts)
        assert "nurse" in joined.lower() or "works as" in joined.lower()
        assert "Caroline" in joined

    def test_extracts_location_move(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        assert result.payload is not None
        facts = " ".join(result.payload.facts).lower()
        assert "seattle" in facts or "moved" in facts

    def test_extracts_preference(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        assert result.payload is not None
        prefs = " ".join(result.payload.preferences).lower()
        assert "hiking" in prefs or "prefer" in prefs or "always" in prefs

    def test_extracts_decision(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        assert result.payload is not None
        decisions = " ".join(result.payload.decisions).lower()
        assert "adopt" in decisions or "decided" in decisions

    def test_extracts_procedure(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        assert result.payload is not None
        procs = " ".join(result.payload.procedures).lower()
        assert "train" in procs or "crate" in procs or "how to" in procs

    def test_extracts_health_fact(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        assert result.payload is not None
        facts = " ".join(result.payload.facts).lower()
        assert "allergic" in facts

    def test_extracts_entities(self):
        result = extract_from_transcript(LOCOMO_TRANSCRIPT, "locomo-1", None)
        names = {e.lower() for e in result.entities}
        assert "caroline" in names
        assert "melanie" in names

    def test_speaker_said_pattern(self):
        text = "Melanie said she graduated from Oregon State University in 2018."
        result = extract_from_transcript(text, "s1", None)
        assert result.payload is not None
        facts = " ".join(result.payload.facts).lower()
        assert "graduated" in facts or "oregon" in facts

    def test_low_signal_returns_none_payload(self):
        result = extract_from_transcript("Hi.\nHow are you?\nGreat.", "s1", None)
        assert result.payload is None

    def test_keyword_fallback_still_works(self):
        text = "We decided to use Redis for caching instead of in-memory storage."
        result = extract_from_transcript(text, "s1", None)
        assert result.payload is not None
        assert any("redis" in d.lower() for d in result.payload.decisions)

    def test_caps_per_category(self):
        decisions = "\n".join(
            f"We decided to use option{i} for the deployment pipeline." for i in range(20)
        )
        facts = "\n".join(
            f"The root cause was a misconfigured timeout in module{i} settings." for i in range(20)
        )
        result = extract_from_transcript(decisions + "\n" + facts, "s1", None)
        assert result.payload is not None
        assert len(result.payload.decisions) <= 6
        assert len(result.payload.facts) <= 12

    def test_deduplicates(self):
        line = "Caroline: I work as a registered nurse at the downtown clinic."
        text = "\n".join([line] * 5)
        result = extract_from_transcript(text, "s1", None)
        assert result.payload is not None
        assert len(result.payload.facts) == 1

    def test_router_wrapper(self):
        r = MemoryRouter._extract_reflection(LOCOMO_TRANSCRIPT, "locomo-1", "/proj/locomo")
        assert r is not None
        assert "locomo" in r.summary

    def test_painted_fact_avoids_duplicate_verb(self):
        text = "Melanie: I painted that lake sunrise in 2022! It's special to me."
        result = extract_from_transcript(text, "s1", None)
        assert result.payload is not None
        facts = " ".join(result.payload.facts)
        assert "painted painted" not in facts.lower()
        assert "lake sunrise" in facts.lower()

    def test_parse_dialogue_turns(self):
        text = "\n".join(
            [
                "[D1:1] Caroline: Hi!",
                "[D1:2] Melanie: I moved to Portland last year.",
                "not a dialogue line",
            ]
        )
        turns = parse_dialogue_turns(text)
        assert len(turns) == 2
        assert turns[0].dia_id == "D1:1"
        assert turns[0].speaker == "Caroline"
        assert turns[1].dia_id == "D1:2"
        assert "Portland" in turns[1].utterance


class TestRoleContradictionCues:
    def test_moved_to_contradicts_lives_in(self):
        old = "Caroline lives in Boston and works nearby."
        new = "Caroline moved to Seattle last month for a new job."
        assert role_conflict_boost(new, old) >= 0.35
        assert contradiction_score(new, old) >= 0.55

    def test_same_city_reinforces_not_contradicts(self):
        old = "Caroline lives in Seattle near the hospital."
        new = "Caroline moved to Seattle last year from Boston."
        boost = role_conflict_boost(new, old)
        assert boost < 0.35

    def test_works_as_contradiction(self):
        old = "Caroline works as a school teacher in Seattle."
        new = "Caroline works as a hospital nurse in Seattle now."
        assert role_conflict_boost(new, old) >= 0.3
        assert contradiction_score(new, old) >= 0.5
