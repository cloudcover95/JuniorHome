# Custom localLLM (this computer)

```
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/enduser_prod.py ./vault "your note"
```

Writes `local_llm.json`: name/runtime/ctx from probe (MLX / CUDA llama.cpp / ARM i2sd / CPU i2sd).
Terraform + inject run against that card so Stock/StoneField inboxes share the same box LLM.
