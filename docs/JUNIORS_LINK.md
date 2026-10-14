# Juniors trit link

`web3node/trit_link_guard.py` links two AntiCon closes (`dev/gaia`, `home/prompt`).
Rate 0.05. released 0. mint false. socket false. layer_one false.

That guard hashes with `validate.make` (prev || pack). Short texts collide there.
`web3node/trit_link_rule.py` is the side-commit link: sha256(prev || pack || text)[:16].
Use the rule for new closes. The guard stays. Nothing is deleted.

```bash
python3 scripts/trit_link_rule_prod.py
```
