# Engine plate — measured vs public (2026-09-13)

Palette: copper on ink. Not the white Gemini notebook slide.

## What this box actually ran
Gaia heightfield 16/32/48 OBJ write. 32² = 1024 verts / 961 faces / ~32 KB ASCII OBJ.
Trit zlib pack vs float32 on n² heights is the 16× story from the BitNet Edge plate (16384 B → ~1 KB on 64²).
UE5 / MLX / CUDA / Blender marked OFF on the JuniorOS live plate. T0 kernel 0 GB live.

## vs public tools (honest)
| Offer | What they sell | Home tonight |
|---|---|---|
| numpy.linalg.svd | Full SVD, any k | Poster: 48×48 ≈ 1 ms retain k=30 @ 0.955 energy — plausible; not a 4k film plate |
| microsoft/BitNet + bitnet.cpp I2_S | Real 1.58 inference | Home is stdlib twin; Metal stays BitNet-mlx |
| llama.cpp GGUF | Q4/Q5/Q8 + I2_S | llama_ready only if JUNIOR_GGUF exists |
| Blender / bpy | Real mesh edit | We emit OBJ + optional py; no .blend shipped |
| Unreal 5 + Datasmith | Editor + Nanite | `gaia_ue5.json` actors only. launch=false |
| Cesium / USGS 3DEP | Real DEM | dem_tile can hit EPQS; automations do **not**. Gaia synth DEM is the offline train set |
| Gemini “70% / 100 Hz / 5×” slide | Marketing macro | Not a measurement on this node. Flip that slide to copper: 45 W intent, not 700 W cluster |

## Train / infer
`web3node/engine_plate.py` writes `~/.juniorhome/gaia_mesh/gaia_train.jsonl` — one line per vertex `{x,y,z,trit}`.
Inference path is Flagstaff + trit_tick, not a 2B checkpoint.
