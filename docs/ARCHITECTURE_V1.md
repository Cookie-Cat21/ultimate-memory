# Ultimate Memory v1 architecture

The v1 direction separates **general memory intelligence** from benchmark-specific answer logic.

## Invariants

1. Production retrieval never accepts a gold benchmark category.
2. Query strategy is inferred from the question by `planner.py`.
3. Structured claims are benchmark-agnostic and stored in atom metadata for backwards compatibility.
4. Current-state conflicts prefer subject/predicate/object semantics before lexical contradiction heuristics.
5. Public benchmark runs must keep raw dialogue input separate from annotation summaries.
6. Answer quality and retrieval quality are measured separately.
7. XL30 remains frozen on `archive/xl30-baseline`.

## Retrieval path

```text
question
  -> query planner
  -> lexical + vector + atom + graph candidates
  -> RRF
  -> salience
  -> semantic/entity/temporal/type reranking
  -> adaptive multi-hop expansion
  -> compact context
  -> answerer
```

## Claim model

Claims live under `AtomicMemory.metadata["claim"]`:

```json
{
  "schema": "ultimate-memory.claim.v1",
  "subject": "Project Koel",
  "predicate": "project.database",
  "object": "PostgreSQL",
  "confidence": 0.82
}
```

This avoids a database migration while allowing future indexed claim columns.

## Evaluation protocol

`evals/run_clean_benchmarks.py` is the credibility benchmark. It:

- ingests only raw conversation sessions,
- ignores LoCoMo observation/event/session summaries,
- never passes the gold question category to `MemoryRouter.answer`,
- contains no media URL/title hints,
- reports evidence recall separately from token-F1.

The historical tuned harness is retained only for regression archaeology.
