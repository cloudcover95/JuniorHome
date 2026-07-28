# JuniorCloud LLC — LLM Portal Status (2026-07-28)

## Multi-Portal Strategy (No Home Lab Required)

Three complementary portals inside JuniorLLM, all running on the shared BitNet 1.58 edge substrate.

### 1. JuniorPortal-K3 (Moonshot Kimi K3)
- Open 2.8T MoE → BitNet 1.58 re-architecture + distillation
- Target edge install 30–80 GB (vs original ~1.45 TB)
- Sparse MoE + SmartExpertOffloader + rigid routing
- Location: JuniorLLM adaptations/kimi_k3/ + junior_bitnet scaffolding

### 2. JuniorLLM-Fable (Claude Fable 5 style)
- Behavioral / architectural adaptation (no public weights)
- Long-horizon agentic reliability + transparent safety classifiers with fallback/refuse
- **NOW WIRED** into IntentRouter (SafetyClassifier runs first on every request)
- Location: JuniorLLM adaptations/fable/

### 3. JuniorGemma-4 (Google Gemma 4)
- Apache 2.0 open source (April 2026)
- Sizes 2B / 4B / 26B MoE / 31B Dense — excellent immediate edge fit
- **BitNet quantization + MLX loader skeleton committed** (fastest interactive path)
- Location: JuniorLLM adaptations/gemma4/

## Production Wiring (Completed)
- IntentRouter updated with Fable SafetyClassifier + multi-portal selection
- Unified multi_portal_production_loop.py covering all three portals
- Agentic pipeline plan documented (AGENTIC_PIPELINES.md)
- All commits additive; no existing files deleted

## Shared Foundation
- BitNet 1.58 ternary core + SVD-Zero + TDA
- Smart offloading / progressive loading
- Production loops + CI modernization + vault security
- Rigid routing for high-stakes (trading, security)
- Offline-first, M4 / van compatible

## Router Flow
1. Fable-style SafetyClassifier
2. Portal selection (Gemma-4 default for speed, Fable for long-horizon, Kimi for long-context/MoE)
3. Existing deterministic tools + memory + fetch preserved
4. Fallback to manifold if needed

*Updated 2026-07-28 — production wiring complete, path to live beta open.*
