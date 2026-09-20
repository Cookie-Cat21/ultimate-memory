# MemEval integration

Ultimate Memory includes an adapter for the independent ProsusAI/MemEval harness.
MemEval standardizes the answer model, embeddings/scoring pipeline and token
tracking across memory systems, making it a stronger public comparison than
cross-vendor self-reported percentages.

## Run

1. Clone ProsusAI/MemEval.
2. Install this repository into the same environment.
3. Copy integrations/memeval/ultimate_memory.py to
   src/agents_memory/systems/ultimate_memory.py in the MemEval checkout.
4. Configure the same API/model settings used by the other systems.
5. Run the MemEval benchmark with system name ultimate_memory.

The adapter deliberately ingests raw dialogue only, receives no gold question
category, runs Ultimate Memory in local-first fallback mode, uses Ultimate
Memory for formation/retrieval, and uses the MemEval-provided llm_model only
for final answer synthesis. This keeps model choice from becoming a hidden
advantage.
