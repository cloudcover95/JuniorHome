# BitNet quant in BitnetCloud

Stdlib AbsMean in `packs/junior_arm_iot/quant.py`.
Same formula as BitNet-mlx: W_q = clip(round(W / mean(|W|)), -1, 1).
5 trits / byte pack.
MMIO sample window → 8-float features → trits → FIELD register.
MLX optional. CPU intent only. Not a Nintendo product.
