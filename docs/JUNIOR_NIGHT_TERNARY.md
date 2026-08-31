# JuniorNightTernary

Original overnight BitNet (not bitnet.cpp).
Implementation lives in JuniorLLM `bitnet_night/`.

```
PYTHONPATH=../JuniorLLM python -c "from bitnet_night.cycle import run_cycle; from pathlib import Path; print(run_cycle(Path('data/night'), 'overnight field score', 64))"
```

Morning brief: MORNING.json with drift EMA + recommendation vocab shared with TDA/FieldCore.
