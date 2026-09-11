# Local infer

```
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/junioros_prod.py ./vault
```

Writes `infer_status.json` (llama ready?, fusion backend, probe) and `flagstaff_ctx.json`.
llama/bitnet.cpp only if GGUF + binary. Else i2sd. Triton JIT is optional and must match CPU y.
