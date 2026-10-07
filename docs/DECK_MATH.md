# Deck math

Per channel: frame of 8, gamma = mean(abs(x)), trit = clip(round(x/gamma), -1, 1), dot = sum(trit * W).
W is (1, 0, -1, 1, 0, -1, 1, 0). No float multiply on the hot path after the pack.
Frame is the channel name until a PCM buffer is present. live false.

```bash
python3 scripts/deck_math_prod.py
```
