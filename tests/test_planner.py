from ultimate_memory.planner import plan_query


def test_infers_multi_hop_without_external_category():
    plan = plan_query("Where does Elena's sister's employer have its headquarters?")
    assert plan.kind == "multi_hop"
    assert plan.hop_depth >= 2


def test_infers_historical_and_superseded():
    plan = plan_query("What database did we use before PostgreSQL?")
    assert plan.kind == "temporal"
    assert plan.include_superseded is True
    assert plan.temporal_mode == "historical"


def test_detects_preference_type():
    plan = plan_query("What package manager do I prefer?")
    assert "preference" in plan.memory_types


def test_as_of_forces_temporal():
    plan = plan_query("Where did the team work?", as_of="2025-01-01T00:00:00+00:00")
    assert plan.kind == "temporal"
    assert plan.temporal_mode == "as_of"


def test_when_question_is_temporal_without_supersession():
    plan = plan_query("When did Elena start her new job?")
    assert plan.kind == "temporal"
    assert plan.include_superseded is False


def test_duration_question_is_temporal():
    plan = plan_query("How long has Marcus worked there?")
    assert plan.kind == "temporal"


def test_collective_both_question_triggers_multi_hop():
    plan = plan_query("What hobbies have both Elena and Marcus enjoyed?")
    assert plan.kind == "multi_hop"
    assert plan.hop_depth >= 2


def test_two_entity_comparison_triggers_multi_hop():
    plan = plan_query("What do Elena and Marcus have in common?")
    assert plan.kind == "multi_hop"


def test_plural_set_query_uses_multi_hop_plan():
    plan = plan_query("Which cities has Jon visited?")
    assert plan.kind == "multi_hop"
    assert plan.hop_depth >= 2


def test_offer_query_uses_multi_evidence_plan():
    plan = plan_query("What does Jon's dance studio offer?")
    assert plan.kind == "multi_hop"


def test_generic_plural_history_query_uses_distributed_plan():
    assert plan_query("What martial arts has Alex done?").kind == "multi_hop"
    assert plan_query("What books has Alex read?").kind == "multi_hop"
    assert plan_query("Where has Alex camped?").kind == "multi_hop"


def test_duration_between_events_uses_multi_evidence_plan():
    plan = plan_query("How long did it take Alex to open the studio?")
    assert plan.kind == "multi_hop"
    assert plan.hop_depth >= 2
    assert any("started" in expansion for expansion in plan.expansions)


def test_distributed_list_query_is_not_a_bridge_walk():
    plan = plan_query("What activities does Melanie partake in?")
    assert plan.kind == "multi_hop"
    assert plan.multi_evidence is True
    assert plan.requires_bridge is False


def test_shared_named_entities_are_multi_evidence_not_unknown_bridge():
    plan = plan_query("Which city have both Jean and John visited?")
    assert plan.multi_evidence is True
    assert plan.requires_bridge is False


def test_possessive_relation_chain_requires_bridge():
    plan = plan_query("Where does Elena's sister's mentor work?")
    assert plan.kind == "multi_hop"
    assert plan.requires_bridge is True


def test_favorite_question_prefers_preference_memory():
    plan = plan_query("What is Gina's favorite style of dance?")
    assert "preference" in plan.memory_types
    assert any("favorite" in expansion for expansion in plan.expansions)


def test_recent_question_is_current_temporal():
    plan = plan_query("What did Melanie paint recently?")
    assert plan.kind == "temporal"
    assert plan.temporal_mode == "current"


def test_in_what_ways_query_is_multi_evidence_not_bridge():
    plan = plan_query("In what ways is Caroline participating in the LGBTQ community?")
    assert plan.multi_evidence is True
    assert plan.requires_bridge is False


def test_group_preference_query_is_multi_evidence():
    plan = plan_query("What do Melanie's kids like?")
    assert plan.multi_evidence is True
    assert plan.requires_bridge is False
