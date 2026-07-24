# Ultimate Memory Benchmark Results

Offline token-F1 vs A-MEM Table 1 (GPT-4o-mini). Mem0/Zep **J** scores need an LLM judge and are not comparable.

## Competitor baselines

| Category | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|
| single_hop | 27.02 | 26.65 | 5.00 | 9.15 |
| multi_hop | 45.85 | 25.52 | 9.68 | 12.60 |
| temporal | 12.14 | 9.15 | 5.56 | 5.31 |
| open_domain | 44.65 | 41.04 | 6.61 | 9.67 |

## Best full LoCoMo-10 (1540 Qs, `google/flan-t5-xl`)

| Category | Ours | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|---:|
| single_hop | **39.82** | 27.02 ✓ | 26.65 ✓ | ✓ | ✓ |
| multi_hop | **27.99** | 45.85 | 25.52 ✓ | ✓ | ✓ |
| temporal | **33.91** | 12.14 ✓ | 9.15 ✓ | ✓ | ✓ |
| open_domain | **33.43** | 44.65 | 41.04 | ✓ | ✓ |

**Scoreboard:** A-MEM **2/4**, MemGPT **3/4**, MemoryBank/ReadAgent **4/4**.

Overall token F1 **36.0** (adversarial skipped).

## Dialog-1 (152 Qs) — full A-MEM sweep

| Model | single | multi | temporal | open |
|---|---:|---:|---:|---:|
| `flan-t5-xl` | 33.5 | **66.2** | 22.5 | **96.2** |

## Synthetic suite

Constraint accuracy **100%**; token F1 **~93%+**.

## Run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite locomo --llm --model google/flan-t5-xl
```
