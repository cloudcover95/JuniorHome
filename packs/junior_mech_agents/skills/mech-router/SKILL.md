---
name: mech-router
description: High-quant ternary router for Grok Super / Plus style agents on JuniorHome. Dispatches scan, CAD, FEA, and stock intents.
---

# Mech router

Stdlib AbsMean 6x16. Same family as FrameForge FFBN / TritARM intent.

| Action | Tree |
|---|---|
| snap_scan | JuniorOmega.sensors |
| emit_glb | JuniorOmega.blender |
| export_step | JuniorOmega.cad |
| fea_flag | JuniorEngrTools.fea |
| stock_intent | JuniorStock |
| halt | JuniorHome |

Rules: load only the child skill for the chosen action. Engineering numbers stay in EngrTools / Omega. Router is logits.
