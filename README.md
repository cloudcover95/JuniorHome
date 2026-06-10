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

All development happens here toward a complete, production-ready codebase.