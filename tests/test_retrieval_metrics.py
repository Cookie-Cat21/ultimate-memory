from ultimate_memory.eval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert 0.0 < ndcg_at_k(ranked, relevant, 4) <= 1.0
