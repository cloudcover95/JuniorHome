# OSai join

Reads `~/.juniorhome/deck/audio_digest.json` if present.
Runs tritquant on the first envelope, else a fixed 8-vector.
Writes `osai_join.json` and appends `gaia_mesh/osai_join.jsonl`.

```bash
python3 scripts/deck_audio_prod.py
python3 scripts/osai_join_prod.py
```

JuniorLLM `ports/deck_trit.py` reads the join. No model pull.
