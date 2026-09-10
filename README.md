# JuniorHome

**Central Sovereign Hub** for the JuniorCloud / BitNet ecosystem.

This is the monorepo-style coordination layer and primary client home.

## Current Structure

- `SwiftApp/` — Hardforked modular SwiftUI client (BitNetEcosystem)
- `docs/` — Architecture, deployment guides, findings
- `deployment/` — Self-host scripts, Docker, k3s manifests (coming)
- `JuniorDrive/` — Robotics, driving simulation, VR/AR AGI
- `benchmarking/` — Platform comparison data
- `ecosystem_tools/` — Supporting scripts (sync_classify, etc.)

## Games on this hub — 2026-09-09

FrameForge is a **local game**, not the home clock.

| Surface | URL | Clock |
|---|---|---|
| FrameForge 2D (itch) | https://cloudcover95.itch.io/frameforge2d | 60 Hz canvas |
| FrameForge2D source | https://github.com/cloudcover95/FrameForge2D | web kernel |
| FrameForge studio + UE | https://github.com/cloudcover95/FrameForge | 120 Hz sim / uncapped render / 20 Hz net |

Drop `FrameForge/juniorhome/FrameForge2D` into `apps/`. Serve loopback `:8765`.
BitNet 8×16 scores CPU intent only. Optional MLX: `cloudcover95/BitNet-mlx`.
Do not boot Unreal on the 45W envelope. Do not let the LLM write stocks or blast.

Unreal 0.4.2 (studio machine): kinematic pawns, Chaos dress-only, ListenIp, Cub local PNG, replay ring.

## Philosophy
- Fully local-first and sovereign
- Ternary x 3.0 + larger model support
- Layered architecture (Core → Intel → Orchestration → Clients)
- No external bloat (Gumroad, heavy Notion dependency)

## Next Goals
- Complete self-host/deployment guides
- Full Swift app integration
- Production-grade orchestration via MCP + BitNet-Intel
- Sim-to-real pipelines in JuniorDrive
- Keep FrameForge 2D on the home node; Unreal stays a studio presenter

All development happens here toward a complete, production-ready codebase.
