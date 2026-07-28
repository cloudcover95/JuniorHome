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
- Location: JuniorLLM adaptations/fable/

### 3. JuniorGemma-4 (Google Gemma 4)
- Apache 2.0 open source (April 2026)
- Sizes 2B / 4B / 26B MoE / 31B Dense — excellent immediate edge fit
- High intelligence-per-parameter, mobile-first, Gemini 3 research lineage
- BitNet quantization + MLX primary path
- Location: JuniorLLM adaptations/gemma4/

## Shared Foundation
- BitNet 1.58 ternary core + SVD-Zero + TDA
- Smart offloading / progressive loading
- Production loops + CI modernization + vault security
- Rigid routing for high-stakes (trading, security)
- Offline-first, M4 / van compatible

## Router
Fable-style SafetyClassifier runs first. Then select or blend portal by task.

## Policy
All work is additive. No existing files or repositories have been deleted or overwritten.

*Updated 2026-07-28 — three portals live and production-oriented toward edge beta.*
