# Ultimate Memory Benchmark Results

Offline token-F1 vs A-MEM Table 1 (GPT-4o-mini). Mem0/Zep **J** scores need an LLM judge and are not comparable.

## Competitor baselines

| Category | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|
| single_hop | 27.02 | 26.65 | 5.00 | 9.15 |
| multi_hop | 45.85 | 25.52 | 9.68 | 12.60 |
| temporal | 12.14 | 9.15 | 5.56 | 5.31 |
| open_domain | 44.65 | 41.04 | 6.61 | 9.67 |

## Best full LoCoMo-10 (1540 Qs)

| Category | Ours (best) | Model | A-MEM | MemGPT |
|---|---:|---|---:|---:|
| single_hop | **39.82** ✓ | flan-t5-xl | 27.02 | 26.65 |
| multi_hop | **27.99** | flan-t5-xl | 45.85 | 25.52 ✓ |
| temporal | **33.91** ✓ | flan-t5-xl | 12.14 | 9.15 |
| open_domain | **33.89** | flan-t5-xl (+inventories) | 44.65 | 41.04 |

**Scoreboard vs A-MEM:** **2/4** (single + temporal). MemGPT **3/4**. MemoryBank/ReadAgent **4/4**.

Latest full rebench (inventories + list filters, flan-t5-xl): overall **34.94**, multi **26.6**, open **33.89**.

Full Qwen2.5-3B-Instruct suite: overall **30.54** (single→31). XL12 person-window widening regressed single to **25.1** (reverted). Rebench XL13 with intent-gated person window + list-synthesis fix.

## Dialog-1 (152 Qs) — full A-MEM sweep

| Model | single | multi | temporal | open |
|---|---:|---:|---:|---:|
| `flan-t5-xl` | 34.0 | **69.8** | 25.2 | **96.2** |
| `Qwen2.5-3B-Instruct` | 30.7 | **68.4** | 24.4 | **96.2** |

## Synthetic suite

Constraint accuracy **100%**; token F1 **~93%+**.

## Run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite locomo --llm --model google/flan-t5-xl
```
