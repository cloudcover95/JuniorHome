# JuniorHome

**Ecosystem Roadmap Progress (June 2026)**

**Current Focus Areas**
- Cross-platform packaging & lean multi-OS support
- Neuromorphic refinements inside BitNet (plasticity training signals)
- JuniorPython automation expansion
- Coordination layer (typed deliverables, reactive handoff, workflow orchestration, Obsidian export)

**BitNet Precision Routing (new)**
- Added lean `BitNetPrecisionRouter`.
- JuniorQuant / trading agents decide *when* to use 1.58-bit vs higher precision.
- BitNet-mlx remains the implementation engine.
- Default is always the efficient 1.58-bit path unless coherence or criticality requires escalation.
- Near-zero overhead design.

The system is designed so that the application layer (JuniorQuant) can make lean, efficient precision decisions while the core BitNet engine stays focused on high-performance ternary math.