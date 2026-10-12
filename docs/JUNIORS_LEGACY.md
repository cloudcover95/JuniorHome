# Legacy rows

Rows without text are legacy. They stay on disk. deleted is 0.
Only a row that passes sha256(prev || pack || text) counts as live.
layer_one false.

```bash
python3 scripts/juniors_legacy_prod.py
```
