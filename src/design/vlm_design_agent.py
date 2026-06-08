# path: src/design/vlm_design_agent.py

"""
VLMDesignAgent

Improved parallel coordination with explicit result merging using plasticity feedback
and GraphMemory deliverables.
"""

from typing import Any, Callable, Dict, List, Optional

import time

try:
    from src.automation.cad_script_generator import CADScriptGenerator
except ImportError:
    CADScriptGenerator = None


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
        cad_generator: Optional[Any] = None,
    ):
        self.bitnet_runner = bitnet_runner
        self.theoretical_math_fn = theoretical_math_fn
        self.plasticity = plasticity_engine
        self.graph_memory = graph_memory
        self.real_data_runner = real_data_runner
        self.vlm_vision_fn = vlm_vision_fn
        self.cad_generator = cad_generator or (CADScriptGenerator() if CADScriptGenerator else None)

        self.design_history: List[DesignState] = []
        self.current_design: Optional[DesignState] = None
        self.efficiency_stats = {
            "iterations_skipped": 0,
            "total_iterations": 0,
            "scripts_generated": 0,
            "quantization_time": 0.0,
            "memory_ops_time": 0.0,
            "vision_analysis_time": 0.0
        }
        self._iteration_start_time = None

    def analyze_design_image(self, image_features: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        if self.vlm_vision_fn:
            result = self.vlm_vision_fn(image_features)
        else:
            result = {
                "shock_wave_strength": image_features.get("edge_density", 0.5),
                "flow_separation_risk": 0.3,
                "structural_stress_hotspots": image_features.get("text_density", 0.2),
                "overall_score": 0.7
            }
        self.efficiency_stats["vision_analysis_time"] += time.time() - start
        return result

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
            "reasoning": "Using memory for efficiency.",
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

    def iterate_design(self, target_goals: Dict[str, float], max_iterations: int = 10, auto_export_scripts: bool = True, parallel_paths: int = 1) -> List[DesignState]:
        self.efficiency_stats["total_iterations"] += 1
        self._iteration_start_time = time.time()

        if not self.current_design:
            self.current_design = DesignState(
                params={"wing_sweep": 35, "nose_shape": "blunt", "length": 30},
                metrics={"drag_coefficient": 0.025, "boom_overpressure": 1.2}
            )

        if self.graph_memory:
            t0 = time.time()
            similar = self.graph_memory.query_similar(
                {"type": "supersonic_design", **self.current_design.params},
                top_k=3
            )
            self.efficiency_stats["memory_ops_time"] += time.time() - t0

            for match in similar:
                match_metrics = match.get("metrics", {})
                if (match_metrics.get("drag_coefficient", 1) < target_goals.get("max_drag", 0.01) and
                    match_metrics.get("boom_overpressure", 1) < target_goals.get("max_boom", 0.6)):
                    self.efficiency_stats["iterations_skipped"] += 1
                    if auto_export_scripts and self.cad_generator:
                        self.cad_generator.export_to_file(match, f"best_design_{int(time.time())}.step", format="step")
                    return [DesignState(params=match.get("params", {}), metrics=match_metrics)]

        iteration_results = []
        best_design = None
        best_score = float('inf')

        for path in range(max(1, parallel_paths)):
            local_best = None
            local_best_score = float('inf')

            for i in range(max_iterations):
                t_vision = time.time()
                vision_analysis = self.analyze_design_image({"edge_density": 0.6, "text_density": 0.4})
                self.efficiency_stats["vision_analysis_time"] += time.time() - t_vision

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
                    t_mem = time.time()
                    self.graph_memory.post_deliverable(
                        {
                            "type": "supersonic_design",
                            "params": new_design.params,
                            "metrics": new_design.metrics
                        },
                        produced_by="VLMDesignAgent",
                        deliverable_type="design",
                        version=i + 1
                    )
                    self.efficiency_stats["memory_ops_time"] += time.time() - t_mem

                # Track best in this parallel path
                score = new_metrics.get("drag_coefficient", 999)
                if score < local_best_score:
                    local_best_score = score
                    local_best = new_design

                if auto_export_scripts and self.cad_generator:
                    if score < 0.015:
                        self.cad_generator.export_to_file(new_design, f"design_iter_{i}.py", format="python_cadquery")
                        self.cad_generator.export_to_file(new_design, f"design_iter_{i}.step", format="step")
                        self.efficiency_stats["scripts_generated"] += 2

                self.design_history.append(new_design)
                iteration_results.append(new_design)
                self.current_design = new_design

                if score < target_goals.get("max_drag", 0.01) and new_metrics.get("boom_overpressure", 1) < target_goals.get("max_boom", 0.6):
                    break

            # Merge: keep the best from this parallel path
            if local_best and local_best_score < best_score:
                best_score = local_best_score
                best_design = local_best

        if best_design:
            self.current_design = best_design

        self.efficiency_stats["last_iteration_time"] = time.time() - self._iteration_start_time
        return iteration_results

    def get_best_design(self) -> Optional[DesignState]:
        if not self.design_history:
            return None
        return min(self.design_history, key=lambda d: d.metrics.get("drag_coefficient", 999))

    def get_efficiency_report(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.efficiency_stats.get("total_iterations", 0),
            "iterations_skipped": self.efficiency_stats.get("iterations_skipped", 0),
            "scripts_generated": self.efficiency_stats.get("scripts_generated", 0),
            "last_iteration_time": self.efficiency_stats.get("last_iteration_time", 0),
            "avg_time_per_iteration": self.efficiency_stats.get("last_iteration_time", 0) / max(self.efficiency_stats.get("total_iterations", 1), 1),
            "quantization_time": self.efficiency_stats.get("quantization_time", 0),
            "memory_ops_time": self.efficiency_stats.get("memory_ops_time", 0),
            "vision_analysis_time": self.efficiency_stats.get("vision_analysis_time", 0)
        }
