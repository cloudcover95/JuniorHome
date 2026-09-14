# SVD inefficiency — production audit

Handshake / dash / markets **do not** run SVD. They pack trit (winsor p95 + I2_S).

Dense `numpy.linalg.svd` was still first-try on every `svd_tick()` (gaia_stack, fieldcore_spine). That is the leftover. 48×48 is cheap; 1024² MLX FieldCore is what we already refused.

Now: `JUNIOR_SVD=auto` (default) uses numpy only if n≤32; else power-k (k≤8). `JUNIOR_SVD=full` is operator-only.

web3node copper plate remains a bench poster, not a Home hot path.
