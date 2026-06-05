# JuniorHome

**Current Ecosystem State**

**Further Modularization of SHEEP Memory**

- Extracted plasticity logic into `PlasticityEngine` (eligibility traces, reward modulation, homeostatic scaling).
- Extracted retrieval logic into `MemoryRetriever` (context-dependent scoring).
- `SHEEPMemory` now composes these smaller, reusable components.

This improves testability, extensibility, and prepares the system for more advanced strategies (different plasticity rules, alternative retrieval scorers, etc.).

The architecture continues to become cleaner while biological fidelity (multi-scale consolidation, sleep-like offline consolidation, plasticity) remains strong.

JuniorMemSys integration path remains open via the `MemoryBackend` abstraction.