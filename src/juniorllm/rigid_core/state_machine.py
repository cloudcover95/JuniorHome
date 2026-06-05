# path: src/juniorllm/rigid_core/state_machine.py

# Refactored to use the new SHEEPMemory module

from ..memory.sheep_memory import SHEEPMemory

# In __init__ (example):
# self.sheep_memory = SHEEPMemory(node_id=node_id)

# The state machine now delegates memory and plasticity operations
# to self.sheep_memory. Plasticity is now eligibility-trace + reward modulated.

# Example usage points:
# - On profile activation or good outcome: self.sheep_memory.update_eligibility_trace(profile)
# - On positive result: self.sheep_memory.apply_plasticity(profile, outcome=..., reward=...)

# This deepens biological fidelity while keeping the core engine clean.
