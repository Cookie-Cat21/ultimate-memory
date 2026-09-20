from ultimate_memory.chain import rank_evidence_chain


def test_chain_reranking_prefers_reachable_target_relation():
    question = "Where does Elena's sister's employer have its headquarters?"
    contexts = [
        {
            "text": "Elena's sister is Fiona.",
            "score": 0.7,
            "provenance": {"entities": ["Elena", "Fiona"]},
        },
        {
            "text": "Fiona works for Northstar Labs.",
            "score": 0.5,
            "provenance": {"entities": ["Fiona", "Northstar Labs"]},
        },
        {
            "text": "Northstar Labs has its headquarters in Toronto.",
            "score": 0.45,
            "provenance": {"entities": ["Northstar Labs", "Toronto"]},
        },
        {
            "text": "Elena likes watercolor painting.",
            "score": 0.8,
            "provenance": {"entities": ["Elena"]},
        },
    ]
    ranked = rank_evidence_chain(question, contexts)
    assert ranked[0]["provenance"]["chain_reachable"] is True
    assert any(
        "headquarters" in item["text"].lower()
        for item in ranked[:2]
    )
