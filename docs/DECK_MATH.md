# Deck math

Per channel: 8-vector, Winsor p95, AbsMean gamma, trit clip {-1,0,1}, pack5.

\[ \gamma = \mathrm{mean}(|x|), \quad q = \mathrm{clip}(\mathrm{round}(x/\gamma), -1, 1) \]

Not a first-letter hash. live false. MX hotswap.

```bash
python3 scripts/deck_math_prod.py
```
