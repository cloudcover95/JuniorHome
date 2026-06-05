# JuniorHome

**Current Ecosystem State**

**STDP Plasticity Investigation & Implementation**

Deepened `PlasticityEngine` with a more explicit STDP-style algorithm:

- Eligibility trace acts as timing signal for pre-synaptic activity recency.
- Clear separation of potentiation (LTP) and depression (LTD) windows.
- Timing-dependent strength: high eligibility → stronger potentiation on positive outcomes.
- Inverse timing for depression on negative outcomes.
- Still includes reward modulation and homeostatic scaling.

This is a practical, abstract approximation of reward-modulated STDP suitable for edge-native systems.

The modular design allows easy future extension (e.g., more precise timing windows, additional neuromodulatory factors, or full three-factor rules).