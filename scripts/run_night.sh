#!/bin/sh
set -e
ROOT=${JUNIORLLM_ROOT:-../JuniorLLM}
export PYTHONPATH="$ROOT"
python3 -m bitnet_night.cycle 2>/dev/null || python3 - << PY
from pathlib import Path
from bitnet_night.cycle import run_cycle
print(run_cycle(Path("data/night"), "JuniorHome overnight BitNet cycle", 96))
PY
