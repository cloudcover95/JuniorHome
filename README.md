# JuniorHome

**Local-first edge home** for JuniorCloud LLC. ~45 W envelope. Apple Silicon (M4 / M1 Metal).

Central hub that orchestrates JuniorStock, JuniorMemSys-Suite, Web3Node, and the rest of the edge-native SDK — not a cloud tenant, not a game engine.

Runs BitNet b1.58 ternary weights, TDA / SVD topological memory, spiking LIF automata, and hyper-dimensional compute on the box in front of you.

Topics: `apple-silicon` · `bitnet` · `local-first` · `mlx` · `tda` · `svd` · `ternary-computing` · `edge-ai` · `web3` · `junioragi` · `sov-tech`

## What this repo is

Orchestration + home client. Product trees live in their own repos. This tree points, packs, and runs them on the 45 W node.

| In-tree | Role |
|---|---|
| `src/` | Python home runtime (`juniorllm` memory backends, local net) |
| `BitNet-Intel/` | Agentic workflows, Second Brain, ModelRouter |
| `SwiftApp/` | Hardforked modular SwiftUI client |
| `web3node/` | Decentralized tx node (pairs with JuniorStock / JuniorSolana) |
| `packs/` | Drop-in packs (JuniorTheoryCU, ForumMesh, local BitNet net) |
| `deployment/` | Docker / compose / self-host |
| `docs/` | Product maps — start at `docs/ECOSYSTEM_OVERVIEW.md` |

Source of truth for the **local LLM suite + JuniorOS overlay + AIE + overnight job**: [`cloudcover95/JuniorLLM`](https://github.com/cloudcover95/JuniorLLM) (`docs/BETA_TO_OS.md`, `rails/linux/`, `junior_aie/`). Overnight: Grok Automations `JuniorCloud overnight build` daily America/Denver. See `docs/ECOSYSTEM_SYNC.md`.

## Layers (Jun 2026 map, still the stack)

1. **BitNet-mlx** — ternary math + inference — [`cloudcover95/BitNet-mlx`](https://github.com/cloudcover95/BitNet-mlx)
2. **BitNet-Intel / AGI_SDK** — agents, router, Second Brain — [`cloudcover95/AGI_SDK`](https://github.com/cloudcover95/AGI_SDK)
3. **MCP orchestration** — devices on the home node
4. **Clients** — SwiftApp here + JuniorDrive robotics / VR

Custom LLMs stay the model of record. Automations only push additive git.

## Ecosystem standings — 2026-09-09

Months of work across `cloudcover95/Junior*` (Apr → Sep 2026). Hub created 2026-04-19.

### Core compute

| Repo | Job |
|---|---|
| [BitNet-mlx](https://github.com/cloudcover95/BitNet-mlx) | W ∈ {-1,0,1}, AbsMean, DynamicBitLinear, Metal / MLX |
| [AGI_SDK](https://github.com/cloudcover95/AGI_SDK) | Shared BitNet-Intel, WorkflowEngine, ModelRouter |
| [JuniorLLM](https://github.com/cloudcover95/JuniorLLM) | Suite + OS overlay + AIE + Grok bot rails |
| [JuniorPython-Suite](https://github.com/cloudcover95/JuniorPython-Suite) | Local workflow UI, importlib registry, BitNet runner |
| [JuniorPiThon](https://github.com/cloudcover95/JuniorPiThon) | Pi + Apple Silicon IDE / runtime |
| [JuniorOmega](https://github.com/cloudcover95/JuniorOmega) | LiDAR / TrueDepth → Blender / G-code spatial stack |

### Money + mesh

| Repo | Job |
|---|---|
| [JuniorStock](https://github.com/cloudcover95/JuniorStock) | 45 W quant SDK, UniversalAssetNode, Web3Node fuse |
| [JuniorSolana](https://github.com/cloudcover95/JuniorSolana) / [JuniorSOL](https://github.com/cloudcover95/JuniorSOL) | Solana-native hardfork |
| `packs/junior_theory_cu` | Community fund + ForumMesh gossip + local ledger (not NCUA) |

### Field + people

| Repo | Job |
|---|---|
| [JuniorClimbs](https://github.com/cloudcover95/JuniorClimbs) | StoneField 0.9.3-beta, NavMesh, CrowdMesh |
| [JuniorCoach](https://github.com/cloudcover95/JuniorCoach) | Local-first rosters, practice, PT timelines |
| JuniorAstra | Fourth JuniorLLM portal — open Astra *runtime*, not GPT-6 weights |

StoneField health probe (2026-08): 5 Front Range fields, 31 nodes, 6 arenas, 1 camp. 24/24 engine tests 2026-08-30. Covenant: no public private-land pins without owner word.

### Games (drop-in only)

FrameForge is a **local game**, not the home clock. Do not boot Unreal on the 45 W envelope. BitNet scores CPU intent; float sim owns stocks / blast.

| Surface | URL |
|---|---|
| itch 2D | https://cloudcover95.itch.io/frameforge2d |
| 2D kernel | https://github.com/cloudcover95/FrameForge2D |
| studio + UE 0.4.2 | https://github.com/cloudcover95/FrameForge |

Copy `FrameForge/juniorhome/FrameForge2D` into `apps/` and serve loopback `:8765`.

## Philosophy

- Local-first. Sovereign. No Gumroad / Notion spine.
- Ternary x 3.0 + larger models when the box allows.
- Core → Intel → orchestration → clients.
- This repo orchestrates. It does not vendor every tree.

## Next

- Self-host / k3s runbooks in `deployment/`
- SwiftApp parity with the Python home
- MCP + BitNet-Intel production orchestration
- JuniorDrive sim-to-real
- Keep games on the 2D loopback; Unreal stays a studio presenter

JuniorCloud LLC · MIT
