from ultimate_memory.reader import ExtractiveReader


def test_yes_no_reader_uses_model_free_fallback():
    reader = ExtractiveReader()
    answer = reader.answer(
        "Does Alice work as a nurse?",
        [{"text": "Alice works as a nurse in Seattle.", "score": 1.0}],
    )
    assert answer in {"Yes", "Alice works as a nurse in Seattle"}


def test_reader_selects_highest_confidence_span_without_loading_transformers():
    reader = ExtractiveReader()
    reader._pipeline = lambda payloads, batch_size=None: [
        {"answer": "Seattle", "score": 0.3},
        {"answer": "Portland", "score": 0.8},
    ]
    answer = reader.answer(
        "Where does Alice live?",
        [
            {"text": "Alice once visited Seattle.", "score": 1.0},
            {"text": "Alice lives in Portland.", "score": 0.9},
        ],
    )
    assert answer == "Portland"
