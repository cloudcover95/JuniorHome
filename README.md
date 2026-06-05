# JuniorHome

**Current Ecosystem State**

**Deepened Plasticity Rules with STDP-style Timing**

The `PlasticityEngine` has been significantly deepened:

- Eligibility trace now explicitly acts as a **timing signal** (STDP approximation).
- High eligibility (recent activity) → stronger potentiation on positive outcomes.
- Low eligibility → weaker effect.
- Negative outcomes trigger depression with inverse timing (classic STDP behavior).
- Still includes reward modulation and homeostatic scaling.

This brings the learning rules much closer to biological reward-modulated STDP while remaining simple and efficient for edge deployment.

The modular `PlasticityEngine` can be easily extended or replaced with more advanced rules in the future.