# JuniorHome

**Current Ecosystem State**

**Started integration path with JuniorMemSys + made SHEEP memory reusable**

- Extracted core SHEEP memory logic into `src/juniorllm/memory/sheep_memory.py` as a standalone `SHEEPMemory` class.
- This module now contains all biologically-inspired memory features (history, reflection, consolidation, replay, plasticity, retrieval).
- The state machine is being refactored to delegate to this module (initial integration points added).

This is the foundation for proper two-tier memory architecture:
- **SHEEPMemory** (in state machine) = fast, online, reasoning-time memory
- **JuniorMemSys-Suite** = persistent, topological long-term memory (future backend)

We have now started actual integration work by making the memory system modular and ready to connect to JuniorMemSys.

Next steps can include fully wiring the delegation in the state machine and beginning to add a MemSys backend interface.