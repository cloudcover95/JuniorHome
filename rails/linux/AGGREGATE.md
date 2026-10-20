# Inference aggregate

Local only. No model pull.

## Bench
JuniorLLM ports/layer_bench.py writes ~/.juniorhome/os/layer_bench.json.

## Aggregate
scripts/home_aggregate.py pulls the latest inference, goldend, and agent status into one tile.
Writes ~/.juniorhome/os/home_inference.json.

Home rails can surface the combined view without running each port.

Additive. MIT.
