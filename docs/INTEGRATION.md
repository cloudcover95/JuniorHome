# JuniorHome suite integration (2026-09-13)

Home orchestrates. It does not vendor every tree.

```
operator / UI :8765|:8771
        |
   JuniorHome  web3node/os_route.py  trit_tick.py  scripts/llm_harness.py
        |
   sibling clone ../JuniorLLM
        |
   ports/home_harness + winsor + Flagstaff + inject + xai_gate
        |
   rails/linux  i2sd :8767   juniorctl (T12 live, T13 next)
```

## How a note travels
1. Home UI or `scripts/llm_harness.py "…"`
2. `os_route` picks surface + port (FF2D / FieldCore / clock / UE5-gated)
3. JuniorLLM Flagstaff votes + terraform + inbox jsonl
4. `trit_tick` / winsor writes vault JSONL
5. Bind stays 127.0.0.1

## In-tree vs sister
| Here | Sister |
|------|--------|
| web3node (clock, route, trit) | JuniorLLM compute + bot rails |
| scripts/*_prod + llm_harness | BitNet-mlx kernels |
| packs/ | JuniorStock / Climbs / Omega / Drive |
| SwiftApp | AGI_SDK agents |
| BitNet-Intel copy | AGI_SDK source of truth |
| deployment/ Docker | do not use host-net for Home tick |

## Live commands
```
python web3node/trit_tick.py
python web3node/os_route.py
python web3node/probe_future.py
python scripts/llm_harness.py "flagstaff fieldcore"
```

## Gaps (honest)
- `docs/ECOSYSTEM_OVERVIEW.md` still dated June 2026
- Docker/compose exists; loopback Python is the real path
- SwiftApp parity with 8771 UI not proven this week
- GGUF / Asahi / UE5 launch = probe only
