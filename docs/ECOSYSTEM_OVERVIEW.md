# Ecosystem overview (2026-09-13)

Home orchestrates. JuniorLLM computes. Do not clone AbsMean 25 times.

## Live path
`os_route` → Flagstaff 6-vote → trit_tick / winsor → 127.0.0.1:8770/8771/8767
`scripts/llm_harness.py` → `JuniorLLM.ports.home_harness`

## Constraints (do not regress)
No model pull. No kernel patch. No Unreal launch (`os_route.launch` false).
GGUF / Asahi / UE5 = probe only (`probe_future.py`).
Docker/compose is leftover; not the Home tick.
Overnight owns T13. LAST_RECEIPT is T12.

## OSai goldens
First species pack: birds (local cards, not eBird).
JuniorLLM `junior_osai/goldens/birds.json` + `ports/osai_goldens.py`.
Expand later (more species). Covenant still applies to nests on private land.

## Layers
1. BitNet-mlx — ternary math
2. JuniorLLM — ports, Flagstaff, rails, OSai card
3. Home — route, tick, packs, SwiftApp
4. Field / stock / omega / drive — sisters
5. FrameForge2D — game, not the clock
