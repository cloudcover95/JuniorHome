# Juniors consensus

Rule: a tip is accepted only if sha256(prev || pack) matches.
A wrong prev is denied. No vote, no peers, no listen.
layer_one stays false.

```bash
python3 scripts/juniors_consensus_prod.py
```
