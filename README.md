# JuniorHome

**Current Ecosystem State**

**Started JuniorMemSys Integration + Backend Abstraction**

- Created `MemoryBackend` abstract interface + `InMemoryBackend` implementation.
- Made `SHEEPMemory` fully backend-aware. It now delegates storage and retrieval to the backend.
- This is the architectural foundation for plugging in a real `JuniorMemSysBackend` later.
- Plasticity rules (with eligibility traces + reward modulation) continue to operate on top of the backend.

The memory system is now cleanly separated:
- **SHEEPMemory** = active reasoning + plasticity logic
- **MemoryBackend** = storage abstraction (ready for JuniorMemSys)

This is a major step toward the desired two-tier biologically-inspired memory architecture for the entire ecosystem.