# JuniorCloud LLC — LLM Portal Status (2026-07-28)

## Dual Portal Strategy inside JuniorLLM

### JuniorLLM-Fable (Claude Fable 5 style)
- **Type**: Behavioral / architectural adaptation (no public weights available for Fable 5)
- **Focus**: Long-horizon agentic reliability, high consistency, transparent safety classifiers with fallback/refuse paths
- **Location**: `JuniorLLM` repo → `adaptations/fable/`
- **Key files**: ARCHITECTURE.md, BEHAVIOR_CARD.md, safety/classifier.py

### JuniorPortal-K3 (Kimi K3 → BitNet)
- **Type**: Full BitNet 1.58-bit MoE re-architecture of open-weight Kimi K3 ideas
- **Focus**: Extreme size reduction (target 30–80 GB edge variant vs original ~1.45 TB), rigid routing, SmartExpertOffloader, MLX-first
- **Location**: Development scaffolding under junior_bitnet/kimi_k3_adaptation/ + pointer in JuniorLLM `adaptations/kimi_k3/`

Both portals are **additive**. No existing files or repos were removed.

They share:
- BitNet core + SVD-Zero + TDA manifold analysis
- Production loops and CI modernization standards
- Vault security / risk assessment posture
- Edge-first (M4 / van) deployment goals

A future model router inside JuniorLLM / JuniorAGI can select or blend portals by task (coding vs quant vs general agentic).

*Updated 2026-07-28 — dual portal live in JuniorLLM.*
