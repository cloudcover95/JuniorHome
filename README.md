# JuniorHome

**Current Ecosystem State**

**Investigation into Neural Plasticity Models for Learning**

Added a lightweight, biologically-inspired plasticity rule (`_apply_plasticity_rule`):

- **Hebbian component**: Strengthens profiles when they are active during positive outcomes (co-activation strengthening).
- **Homeostatic component**: Applies mild normalization to prevent runaway performance growth (inspired by synaptic scaling).

This rule is now optionally used in `update_from_manifold` when the system detects strong manifold states, and can be extended for use in replay, consolidation, or adapter training.

The SHEEP memory + plasticity system is evolving toward a more complete, neuroscience-grounded learning architecture suitable for sovereign ternary inference on edge hardware.

Next natural step: Integrate these plasticity rules more deeply with the LowRankAdapter training loop and/or port the memory + plasticity layer into JuniorMemSys for persistent, topological learning.