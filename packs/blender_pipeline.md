# blender_pipeline pack

Umbrella pointer. Implementation lives in JuniorOmega `blender/jc_blender.py` + `blender/headless_worker.py` next to `metal_render.py`.

Jobs: `omega-lidar`, `llm-fieldcore`, `agi-capsule`, `frameforge-display`.
LOD trit: -1 perf / 0 balanced / +1 detailed.
CI never requires Blender. UnrealEditor does not boot on the 45 W node.

```
python3 blender/jc_blender.py omega-lidar blender_out 0
```

JuniorCloud LLC.
