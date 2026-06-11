from ultimate_memory.chunking import chunk_text


def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("hello", chunk_chars=20, overlap=5) == ["hello"]


def test_chunk_text_splits_long_text():
    chunks = chunk_text("a" * 50 + ". " + "b" * 50, chunk_chars=40, overlap=5)
    assert len(chunks) > 1
    assert all(chunks)

