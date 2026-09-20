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


def test_temporal_reader_uses_structured_reasoner_without_model():
    reader = ExtractiveReader()
    contexts = [
        {
            "text": "---\nsession_date: 4 February, 2023\n---\nJon: My group is performing at the festival this month.",
            "memory_type": "log",
            "score": 0.9,
        }
    ]
    answer = reader.answer("When is Jon's group performing at a festival?", contexts)
    assert "February" in answer and "2023" in answer


def test_distributed_reader_can_combine_multiple_memories_without_model():
    reader = ExtractiveReader()
    contexts = [
        {"text": "Jon: I visited Paris last winter.", "score": 0.9},
        {"text": "Jon: I took a trip to Rome this summer.", "score": 0.8},
    ]
    answer = reader.answer("Which cities has Jon visited?", contexts)
    assert "Paris" in answer and "Rome" in answer
