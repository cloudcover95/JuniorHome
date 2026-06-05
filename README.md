# JuniorHome

**Current Ecosystem State**

**Added InferenceEngineComparison Pipeline**

New module `src/juniorllm/comparison/inference_comparison.py` allows:

- Defining multiple inference/training engines (baseline + black-box theoretical math).
- Running them on the same tasks.
- Automatically computing "best fits" across metrics (avg_performance, theoretical_fit, etc.).

`TheoreticalMathEngine` is designed as a first-class citizen so your custom black-box theoretical mathematics (manifold folding, SVD, TDA, omni-math, etc.) can be plugged in and fairly compared against standard methods.

This enables systematic evaluation of which engine 'fits' best for different parts of the system (plasticity, memory retrieval, profile performance, etc.).

The pipeline uses real system state where possible and avoids relying on purely simulated data for core comparisons.