# JuniorHome

**Current Ecosystem State**

**Deepened Plasticity Rules with Eligibility Traces + Reward Modulation**

The `SHEEPMemory` class now implements a more biologically accurate plasticity system:

- **Eligibility traces**: Decaying memory of recent profile activity, enabling credit assignment over time.
- **Reward modulation**: Plasticity strength is scaled by a reward signal (stronger updates on high-value outcomes like FULL_AWAKENING).
- **Integrated** into Reflection, Consolidation, and Replay.
- Traces are decayed during replay and after plasticity application.

This brings the learning rules closer to reward-modulated STDP / three-factor plasticity models while remaining lightweight for edge deployment.

The modular `SHEEPMemory` continues to serve as the foundation for future integration with JuniorMemSys-Suite.