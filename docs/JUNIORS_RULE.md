# Close rule

Hash is sha256(prev || pack || text)[:16].
Changing the text fails the old hash. Trit body alone is not the fingerprint.
layer_one false.

```bash
python3 scripts/juniors_rule_prod.py
```
