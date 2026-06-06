# JuniorHome

**Current Ecosystem State**

**Hebbian Learning Dynamics now a modular blackbox**

- Extracted `HebbianStructuralModule` as a swappable blackbox component.
- PlasticityEngine can now accept custom Hebbian rules or future theoretical variants.
- Added `set_hardware_backend()` hook for future Neo chip / CUDA / hardware-agnostic routing.

This increases modularity for both biological learning rules and hardware backends while keeping the core STDP + reward system intact.

Ecosystem continues to evolve toward flexible, sovereign, multi-chip capable inference + memory architecture.