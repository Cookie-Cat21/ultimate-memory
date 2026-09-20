from ultimate_memory.propositions import ground_speaker_reference, turn_propositions


def test_ground_speaker_reference_rewrites_first_person():
    text = ground_speaker_reference("Caroline", "I am a designer and my sister lives in Paris.")
    assert "Caroline is a designer" in text
    assert "Caroline's sister" in text


def test_turn_propositions_splits_sentences_and_semicolons():
    props = turn_propositions(
        "Caroline",
        "I moved to Austin. My sister lives in Paris; I prefer tea.",
    )
    assert any("Caroline moved to Austin" in p for p in props)
    assert any("Caroline's sister lives in Paris" in p for p in props)
    assert any("Caroline prefer tea" in p or "Caroline prefers tea" in p for p in props)


def test_turn_propositions_deduplicates():
    props = turn_propositions("Alice", "I like tea. I like tea.")
    assert len(props) == 1
