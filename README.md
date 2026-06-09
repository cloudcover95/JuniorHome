# JuniorHome

**Ecosystem Development — Balanced High-Quality Iteration**

**JuniorPython automation expanded**
- Added `blender_python` and `gltf_export` formats.
- `batch_generate_artifacts()` now supports custom format lists and is more resilient.

**Cross-platform packaging**
- `pyproject.toml` now has `apple-silicon` and `cpu` extras.
- Makefile has `install-apple`, `install-cpu`, and better platform detection.

**Deeper spiking / neuromorphic in PlasticityEngine**
- `SpikingPlasticityModule` improved with better timing windows, homeostatic regulation, and spike counting.
- Training signals now richer for routing and historical use.

**Workflow orchestration & reactive subscriptions**
- `get_ready_tasks()` and `get_next_work_items()` remain strong.
- Reactive type subscriptions continue to power automatic handoff (VLMDesignAgent → CADScriptGenerator).

All changes maintain lean, sovereign, BitNet-native principles with self-tests.