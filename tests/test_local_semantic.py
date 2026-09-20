from ultimate_memory.local_semantic import LocalSemanticIndex, cosine
from ultimate_memory.models import AtomicMemory, MemoryType


def fake_embed(texts):
    vectors = []
    for text in texts:
        lower = text.lower()
        vectors.append([
            1.0 if "dance" in lower or "studio" in lower else 0.0,
            1.0 if "database" in lower or "postgres" in lower else 0.0,
        ])
    return vectors


def test_cosine_identity():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_semantic_index_ranks_related_atom():
    index = LocalSemanticIndex(fake_embed)
    atoms = [
        AtomicMemory(text="Project uses PostgreSQL.", memory_type=MemoryType.FACT),
        AtomicMemory(text="The dance studio offers classes.", memory_type=MemoryType.FACT),
    ]
    results = index.search("What does the studio offer?", atoms, limit=2)
    assert "dance studio" in results[0].text.lower()
    assert results[0].provenance["source"] == "local-semantic"
