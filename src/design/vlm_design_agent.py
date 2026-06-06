# path: src/design/vlm_design_agent.py

"""
VLMDesignAgent

Now uses HybridSqueezeBitNetQuantizer for quantizing internal design features and model-related tensors when available.

This brings sensitivity-aware + BitNet ternary quantization directly into the design iteration loop.
"""

from typing import Any, Callable, Dict, List, Optional

import time

try:
    from src.quantization.hybrid_squeeze_bitnet import HybridSqueezeBitNetQuantizer
except ImportError:
    HybridSqueezeBitNetQuantizer = None


class DesignState:
    def __init__(self, params: Dict[str, Any], metrics: Dict[str, float] = None):
        self.params = params
        self.metrics = metrics or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }


class VLMDesignAgent:
    def __init__(
        self,
        bitnet_runner=None,
        theoretical_math_fn: Optional[Callable] = None,
        plasticity_engine=None,
        graph_memory: Optional[Any] = None,
        real_data_runner=None,
        vlm_vision_fn: Optional[Callable] = None,
    ):
        self.bitnet_runner = bitnet_runner
        self.theoretical_math_fn = theoretical_math_fn
        self.plasticity = plasticity_engine
        self.graph_memory = graph_memory
        self.real_data_runner = real_data_runner
        self.vlm_vision_fn = vlm_vision_fn
        self.hybrid_quantizer = HybridSqueezeBitNetQuantizer() if HybridSqueezeBitNetQuantizer else None

        self.design_history: List[DesignState] = []
        self.current_design: Optional[DesignState] = None
        self.efficiency_stats = {"iterations_skipped": 0, "total_iterations": 0}

    def analyze_design_image(self, image_features: Dict[str, Any]) -> Dict[str, Any]:
        if self.vlm_vision_fn:
            return self.vlm_vision_fn(image_features)
        return {
            "shock_wave_strength": image_features.get("edge_density", 0.5),
            "flow_separation_risk": 0.3,
            "structural_stress_hotspots": image_features.get("text_density", 0.2),
            "overall_score": 0.7
        }

    def propose_design_changes(self, current_state: DesignState, target_goals: Dict[str, float]) -> Dict[str, Any]:
        context = {}
        if self.graph_memory:
            context = self.graph_memory.get_context_for_agent(
                {"type": "supersonic_design", **current_state.params}
            )

        prompt_context = {
            "current_params": current_state.params,
            "current_metrics": current_state.metrics,
            "goals": target_goals,
            "history_length": len(self.design_history),
            "graph_context": context
        }

        if self.theoretical_math_fn:
            try:
                proposal = self.theoretical_math_fn(prompt_context, outcome=1.0)
                if isinstance(proposal, dict):
                    return proposal
            except Exception as e:
                print(f"[VLMDesignAgent] Theoretical math error: {e}")

        if self.bitnet_runner:
            result = self.bitnet_runner.run_inference(prompt_context)
            return {"suggested_changes": result.get("output", {}), "confidence": 0.75}

        return {
            "suggested_changes": {
                "increase_wing_sweep": 2.5,
                "refine_nose_shape": "sharper_ogive",
                "add_strake": True
            },
            "reasoning": "Using memory and hybrid quantization for efficiency.",
            "confidence": 0.65
        }

    def evaluate_design(self, design: DesignState, simulation_results: Dict[str, Any] = None) -> Dict[str, float]:
        if simulation_results:
            return {
                "drag_coefficient": simulation_results.get("drag", 0.015),
                "boom_overpressure": simulation_results.get("boom", 0.9),
                "structural_safety_factor": simulation_results.get("safety", 1.5)
            }

        return {
            "drag_coefficient": design.metrics.get("drag_coefficient", 0.02) * 0.95,
            "boom_overpressure": design.metrics.get("boom_overpressure", 1.0) * 0.92,
            "structural_safety_factor": 1.8
        }

    def learn_from_design(self, design: DesignState, outcome: float):
        if not self.plasticity:
            return
        profile = "supersonic_design"
        self.plasticity.update_eligibility_trace(profile, strength=abs(outcome))
        self.plasticity.apply(
            performance={"supersonic_design": 0},
            lifecycle={},
            profile=profile,
            outcome=outcome
        )
        if hasattr(self.plasticity, "adapt_meta_plasticity"):
            self.plasticity.adapt_meta_plasticity(design.metrics.get("drag_coefficient", 0.5))

    def _quantize_design_tensor(self, tensor: Any) -> Any:
        """Quantize internal design-related tensors using hybrid quantizer."""
        if self.hybrid_quantizer and tensor is not None:
            try:
                return self.hybrid_quantizer.quantize(tensor)
            except Exception as e:
                print(f"[VLMDesignAgent] Quantization error: {e}")
        return tensor

    def iterate_design(self, target_goals: Dict[str, float], max_iterations: int = 10) -> List[DesignState]:
        self.efficiency_stats["total_iterations"] += 1

        if not self.current_design:
            self.current_design = DesignState(
                params={"wing_sweep": 35, "nose_shape": "blunt", "length": 30},
                metrics={"drag_coefficient": 0.025, "boom_overpressure": 1.2}
            )

        if self.graph_memory:
            similar = self.graph_memory.query_similar(
                {"type": "supersonic_design", **self.current_design.params},
                top_k=3
            )
            for match in similar:
                match_metrics = match.get("metrics", {})
                if (match_metrics.get("drag_coefficient", 1) < target_goals.get("max_drag", 0.01) and
                    match_metrics.get("boom_overpressure", 1) < target_goals.get("max_boom", 0.6)):
                    self.efficiency_stats["iterations_skipped"] += 1
                    return [DesignState(params=match.get("params", {}), metrics=match_metrics)]

        iteration_results = []

        for i in range(max_iterations):
            vision_analysis = self.analyze_design_image({"edge_density": 0.6, "text_density": 0.4})

            # Quantize vision features if possible
            if isinstance(vision_analysis, dict):
                for key in vision_analysis:
                    if isinstance(vision_analysis[key], (list, tuple)) or hasattr(vision_analysis[key], "__array__"):
                        vision_analysis[key] = self._quantize_design_tensor(vision_analysis[key])

            proposal = self.propose_design_changes(self.current_design, target_goals)

            new_params = self.current_design.params.copy()
            for change, value in proposal.get("suggested_changes", {}).items():
                if change.startswith("increase_"):
                    key = change.replace("increase_", "")
                    new_params[key] = new_params.get(key, 0) + value

            new_design = DesignState(params=new_params)

            new_metrics = self.evaluate_design(new_design)
            new_design.metrics = new_metrics

            outcome = 1.0 if new_metrics.get("drag_coefficient", 1) < self.current_design.metrics.get("drag_coefficient", 1) else -0.5
            self.learn_from_design(new_design, outcome)

            if self.graph_memory:
                try:
                    self.graph_memory.store_pattern({
                        "type": "supersonic_design",
                        "params": new_design.params,
                        "metrics": new_design.metrics
                    })
                except Exception as e:
                    print(f"[VLMDesignAgent] GraphMemory error: {e}")

            self.design_history.append(new_design)
            iteration_results.append(new_design)
            self.current_design = new_design

            if (new_metrics.get("drag_coefficient", 1) < target_goals.get("max_drag", 0.01) and
                new_metrics.get("boom_overpressure", 1) < target_goals.get("max_boom", 0.6)):
                break

        return iteration_results

    def get_best_design(self) -> Optional[DesignState]:
        if not self.design_history:
            return None
        return min(self.design_history, key=lambda d: d.metrics.get("drag_coefficient", 999))

    def get_efficiency_stats(self) -> Dict[str, Any]:
        return self.efficiency_stats.copy()
