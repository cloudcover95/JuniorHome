# Juniors ledger

Each row stores prev, pack hex, and hash. walk recomputes sha256(prev || pack).
90 closes are benched in memory. layer_one false. peers 0.

```bash
python3 scripts/juniors_ledger_prod.py
```
