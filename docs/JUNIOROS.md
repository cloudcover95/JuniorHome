# JuniorOS hook

```
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/junioros_prod.py ./vault
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/junioros_prod.py ./vault --overlay
```

Writes `layer1_lock.json` + `flagstaff_ctx.json`. Overlay tarball is optional.
Vendor kernel. I2_S userspace. No custom vmlinuz.
