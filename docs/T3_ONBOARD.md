# T3 onboard

Drop a local `*.gguf` in `~/.juniorhome/models/` or set `JUNIOR_GGUF`.
Do not curl a 2B BitNet from Home.

```bash
mkdir -p ~/.juniorhome/models
python scripts/t3_gguf_prod.py
python scripts/gguf_quants_prod.py
```

FieldCore stays 32²/48² jsonl under `~/.juniorhome/gaia_mesh/`. No parquet.
Ten engines remain the six runtimes + catalog labels, not ten new .so files.
