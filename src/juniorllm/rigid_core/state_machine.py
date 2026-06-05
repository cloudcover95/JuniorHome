# path: src/juniorllm/rigid_core/state_machine.py

# ... (imports and other code remain)

from ..memory.sheep_memory import SHEEPMemory

# In __init__:
# self.sheep_memory = SHEEPMemory(node_id=node_id)

# Then replace internal _sheep_history, _sheep_consolidated_insights, etc.
# with delegation to self.sheep_memory where appropriate.

# Example integration points (to be fully wired in next iterations):
# - record_awakening -> self.sheep_memory.record_awakening(...)
# - reflect -> self.sheep_memory.reflect_on_recent()
# - consolidate -> self.sheep_memory.consolidate()
# - replay -> self.sheep_memory.replay_and_consolidate()
# - retrieve -> self.sheep_memory.retrieve_relevant(...)
# - plasticity -> self.sheep_memory.apply_plasticity(...)

# This refactor prepares the SHEEP memory system to be easily
# swapped with or backed by a JuniorMemSys implementation.
