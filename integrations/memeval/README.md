# Ultimate Memory on ProsusAI MemEval

ProsusAI/MemEval is the primary external fair-comparison gate for Ultimate Memory.
It standardizes the answer LLM, embedding model, scoring pipeline, and token accounting
across memory systems.

At the time this integration was added, MemEval's published LoCoMo leader is PropMem
at 0.605 token-F1 and 0.823 judge score.

## Run head-to-head

Clone MemEval next to Ultimate Memory, install both projects, and copy the adapter:

```bash
git clone https://github.com/ProsusAI/MemEval.git
cd MemEval
uv sync --all-extras
uv pip install -e ../ultimate-memory
cp ../ultimate-memory/integrations/memeval/ultimate_memory.py \
  src/agents_memory/systems/ultimate_memory.py
```

Set `OPENAI_API_KEY`, then run one conversation first:

```bash
uv run python scripts/run_full_benchmark.py \
  --systems ultimate_memory,propmem \
  --num-samples 1 \
  --llm-model gpt-4.1-mini \
  --skip-judge
```

For the full externally comparable LoCoMo run:

```bash
uv run python scripts/run_full_benchmark.py \
  --systems ultimate_memory,propmem \
  --num-samples 10 \
  --llm-model gpt-4.1-mini
```

Do not claim a leaderboard position from Ultimate Memory's internal clean harness.
Use the MemEval result for cross-system claims.

## Fairness

The adapter:

- ingests only raw conversation sessions;
- never receives the gold question category;
- uses MemEval's supplied answer model;
- uses `text-embedding-3-small` for semantic fallback;
- leaves MemEval's scoring and judge code untouched;
- counts answer-LLM tokens through MemEval's normal OpenAI instrumentation.
