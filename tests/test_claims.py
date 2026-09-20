from ultimate_memory.claims import ensure_claim_metadata, structured_conflict_score
from ultimate_memory.models import AtomicMemory, MemoryType


def atom(text: str) -> AtomicMemory:
    return AtomicMemory(text=text, memory_type=MemoryType.FACT)


def test_location_claim_conflict_is_structural():
    old = atom("Carol lives in New York City.")
    new = atom("Carol moved to Austin.")
    ensure_claim_metadata(old)
    ensure_claim_metadata(new)
    assert old.metadata["claim"]["predicate"] == "location.current"
    assert new.metadata["claim"]["predicate"] == "location.current"
    assert structured_conflict_score(new, old) >= 0.8


def test_different_subjects_do_not_conflict():
    assert structured_conflict_score(
        atom("Carol lives in New York City."),
        atom("Gabe lives in Austin."),
    ) == 0.0


def test_project_database_claim():
    memory = atom("Project Koel uses PostgreSQL.")
    claim = ensure_claim_metadata(memory)
    assert claim is not None
    assert claim.predicate == "project.database"
    assert claim.object == "PostgreSQL"
