# web3node ternary slice (2026-09-13)

Additive. Does not rewrite `svd_metal.py`.

| File | Role |
|---|---|
| `web3node/svd_metal.py` | Live Metal SVD. Unchanged. |
| `web3node/svd_retain.py` | Rank-k energy retention. |
| `web3node/ternary_tda.py` | AbsMean ternary + drift + FrameForge CPU logits. |
| `web3node/svd_residual_tick.py` | SVD reconstruct, residual into `TernaryTDAMesh.tick`. |
| `web3node/vietoris_rips.py` | Lean VR 1-skeleton. giotto stays in `tda_mesh.py`. |
| `web3node/trit_cache.py` | 2-bit trit pack + zlib; zstd when installed. |
| `web3node/bench_web3node.py` | Local bench. |
| `deployment/Dockerfile.web3node-harness` | Thin numpy harness toward stocksnode. |

Knockback stays in FrameForge. BitNet / ternary scores intent only.
