# Suite tick

```bash
python3 scripts/suite_tick_prod.py
```

Scans `~/.juniorhome/deck/inbox`, then writes `osai_join.json`.
JuniorLLM `scripts/deck_trit_prod.py` reads it and may append a receipt.
