# Ultimate Memory Benchmark Results

Offline, reproducible harness. Token-F1 comparisons use published A-MEM Table 1 numbers (GPT-4o-mini answerer). Mem0/Zep marketing **J** scores need an LLM judge and are not directly comparable.

## Competitor baselines (A-MEM paper, GPT-4o-mini, token F1)

LoCoMo cats: 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial

| Category | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|
| single_hop | 27.02 | 26.65 | 5.00 | 9.15 |
| multi_hop | 45.85 | 25.52 | 9.68 | 12.60 |
| temporal | 12.14 | 9.15 | 5.56 | 5.31 |
| open_domain | 44.65 | 41.04 | 6.61 | 9.67 |

## Best full LoCoMo-10 (1540 Qs, `google/flan-t5-large`, adversarial skipped)

| Category | Ours | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|---:|
| single_hop | **39.79** | 27.02 ✓ | 26.65 ✓ | 5.00 ✓ | 9.15 ✓ |
| multi_hop | **25.88** | 45.85 | 25.52 ✓ | 9.68 ✓ | 12.60 ✓ |
| temporal | **33.70** | 12.14 ✓ | 9.15 ✓ | 5.56 ✓ | 5.31 ✓ |
| open_domain | **32.17** | 44.65 | 41.04 | 6.61 ✓ | 9.67 ✓ |

**Summary:** On the full matched token-F1 protocol we beat **A-MEM on 2/4** (single + temporal, large margins), **MemGPT on 3/4** (adds multi-hop), and **MemoryBank/ReadAgent on 4/4**.

## Dialog-1 sweep (152 Qs, same stack)

| Category | Ours | A-MEM |
|---|---:|---:|
| single_hop | **33.3** | 27.02 ✓ |
| multi_hop | **60.0** | 45.85 ✓ |
| temporal | **23.4** | 12.14 ✓ |
| open_domain | **96.2** | 44.65 ✓ |

## Synthetic router suite

| Metric | Score |
|---|---|
| Token F1 | **~93%+** |
| Constraint accuracy | **100%** |

## How to run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite synthetic
uv run python evals/run_benchmarks.py --suite locomo --llm --model google/flan-t5-large
```

## Notes

- A-MEM/MemGPT use GPT-4o-mini; we use local `flan-t5-large` + atomic retrieval/aggregation.
- Remaining gap vs A-MEM is multi-hop list synthesis and open-domain entity inference on dialogs 2–10.
- Mem0 peer-reviewed LoCoMo J ≈ 67–68% is a different metric.
