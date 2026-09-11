# Layer-1 IQ

JuniorLLM `ports/layer1_iq.py` walks probe → i2s → teqp → ports → terraform.
Locks tiers in `vault/layer1_lock.json`. Second cycle does not unlock.
Language goes through `ports/terraform.py` + local `pick_eos`, not a cloud model.

```
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/layer1_prod.py ./vault/layer1_lock.json
```
