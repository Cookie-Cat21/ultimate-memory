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
