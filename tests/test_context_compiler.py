from ultimate_memory.context_compiler import compile_context_packet


def item(text, score=1.0, session="s1"):
    return {
        "id": text[:8],
        "text": text,
        "score": score,
        "provenance": {"session_id": session},
    }


def test_near_duplicates_are_removed():
    contexts = [
        item("Caroline has a guinea pig named Clover."),
        item("Caroline has a guinea pig named Clover!"),
        item("Jon runs a dance studio.", session="s2"),
    ]
    packet = compile_context_packet(contexts)
    assert len(packet) == 2


def test_budget_is_respected():
    contexts = [
        item("alpha " * 20, session="a"),
        item("beta " * 20, session="b"),
    ]
    packet = compile_context_packet(contexts, max_chars=130)
    assert sum(len(x["text"]) + 2 for x in packet) <= 132


def test_source_diversity_cap():
    contexts = [
        item(f"distinct fact number {i} with unique token x{i}", session="same")
        for i in range(10)
    ]
    packet = compile_context_packet(contexts, max_per_source=3)
    assert len(packet) == 3
