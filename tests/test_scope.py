from ultimate_memory.scope import MemoryScope, scope_boost


def test_global_scope_matches_project():
    assert MemoryScope().matches(MemoryScope(project="koel"))


def test_other_project_does_not_match():
    assert not MemoryScope(project="alpha").matches(MemoryScope(project="koel"))


def test_specific_scope_gets_boost():
    requested = MemoryScope(project="koel", repository="app", branch="main")
    assert scope_boost(MemoryScope(project="koel"), requested) > 0
    assert scope_boost(MemoryScope(project="other"), requested) < 0
