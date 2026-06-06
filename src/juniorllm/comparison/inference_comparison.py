# path: src/juniorllm/comparison/inference_comparison.py

"""
InferenceEngineComparison

VisionTextEngine now includes explicit iPhone baseline comparison support
for benchmarking BitNet vs modern Apple on-device capabilities.

Also prepares structured output for JuniorMemSys topological storage.
"""

import time
import statistics
from typing import Any, Callable, Dict, List, Optional


class InferenceEngine:
    def __init__(self, name: str):
        self.name = name

    def train_step(self, state: Dict[str, Any], outcome: float) -> Dict[str, Any]:
        raise NotImplementedError

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


class BaselinePlasticityEngine(InferenceEngine):
    def __init__(self):
        super().__init__("baseline_plasticity")

    def train_step(self, state: Dict[str, Any], outcome: float) -> Dict[str, Any]:
        profile = state.get("active_profile", "general")
        performance = state.get("performance", {})
        if profile not in performance:
            performance[profile] = 0.0
        performance[profile] += 0.01 * outcome
        state["performance"] = performance
        return state

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        perf = state.get("performance", {})
        if not perf:
            return {"avg_performance": 0.0}
        return {"avg_performance": sum(perf.values()) / len(perf)}


class TheoreticalMathEngine(InferenceEngine):
    def __init__(self, name: str = "theoretical_math"):
        super().__init__(name)
        self.theoretical_math_fn: Optional[Callable] = None

    def set_theoretical_math(self, fn: Callable):
        self.theoretical_math_fn = fn

    def train_step(self, state: Dict[str, Any], outcome: float) -> Dict[str, Any]:
        profile = state.get("active_profile", "general")
        performance = state.get("performance", {})

        if self.theoretical_math_fn is not None:
            try:
                math_result = self.theoretical_math_fn(state, outcome)
                if isinstance(math_result, dict) and "updated_performance" in math_result:
                    performance.update(math_result["updated_performance"])
            except Exception as e:
                print(f"[TheoreticalMathEngine] Error: {e}")
        else:
            if profile not in performance:
                performance[profile] = 0.0
            performance[profile] += 0.015 * outcome * 1.2

        state["performance"] = performance
        return state

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        perf = state.get("performance", {})
        if not perf:
            return {"avg_performance": 0.0, "theoretical_fit": 0.0}
        avg = sum(perf.values()) / len(perf)
        return {"avg_performance": avg, "theoretical_fit": avg * 1.1}


class BitNetVoiceEngine(InferenceEngine):
    def __init__(self, name: str = "bitnet_voice"):
        super().__init__(name)
        self.model_fn: Optional[Callable] = None

    def set_model(self, fn: Callable):
        self.model_fn = fn

    def train_step(self, state: Dict[str, Any], outcome: float) -> Dict[str, Any]:
        features = state.get("audio_features", {})
        performance = state.get("performance", {})

        if self.model_fn is not None:
            try:
                score = self.model_fn(features)
                performance["voice_score"] = float(score)
            except Exception as e:
                print(f"[BitNetVoiceEngine] Model error: {e}")
        else:
            energy = features.get("energy", 0.0)
            performance["voice_score"] = min(1.0, energy * 8)

        state["performance"] = performance
        return state

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        perf = state.get("performance", {})
        score = perf.get("voice_score", 0.0)
        return {
            "avg_performance": score,
            "theoretical_fit": score,
            "is_human": 1.0 if score > 0.2 else 0.0
        }


class VisionTextEngine(InferenceEngine):
    """
    BitNet-native engine for Instagram story zoom tag inference.

    Supports direct benchmarking against modern iPhone capabilities.
    Outputs structured data ready for JuniorMemSys topological storage.
    """

    def __init__(self, name: str = "vision_text_tag_bitnet"):
        super().__init__(name)
        self.vision_fn: Optional[Callable] = None
        self.theoretical_math_fn: Optional[Callable] = None
        self.quantize_features: bool = True

    def set_vision_model(self, fn: Callable):
        self.vision_fn = fn

    def set_theoretical_math(self, fn: Callable):
        self.theoretical_math_fn = fn

    def enable_bitnet_quantization(self, enabled: bool = True):
        self.quantize_features = enabled

    def train_step(self, state: Dict[str, Any], outcome: float) -> Dict[str, Any]:
        image_features = dict(state.get("image_features", {}))
        frame_info = state.get("frame_info", {})
        performance = state.get("performance", {})

        if self.quantize_features:
            for key in list(image_features.keys()):
                if isinstance(image_features[key], (int, float)):
                    val = float(image_features[key])
                    if val > 0.5:
                        image_features[key] = 1.0
                    elif val < -0.5:
                        image_features[key] = -1.0
                    else:
                        image_features[key] = 0.0

        detected_tags: List[str] = []

        if self.vision_fn is not None:
            try:
                detected_tags = self.vision_fn(image_features)
            except Exception as e:
                print(f"[VisionTextEngine] Vision error: {e}")

        if self.theoretical_math_fn is not None:
            try:
                reasoning = self.theoretical_math_fn({
                    "detected_tags": detected_tags,
                    "frame_info": frame_info,
                    "is_zoomed": frame_info.get("zoom_level", 1.0) > 1.5,
                    "quantized_features": image_features
                }, outcome)
                if isinstance(reasoning, dict):
                    performance.update(reasoning)
            except Exception as e:
                print(f"[VisionTextEngine] Theoretical math error: {e}")
        else:
            if image_features.get("text_density", 0) > 0.3:
                performance["detected_account_tags"] = ["@detected"]

        state["performance"] = performance
        state["detected_tags"] = detected_tags
        state["quantized_image_features"] = image_features
        return state

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        perf = state.get("performance", {})
        tags = state.get("detected_tags", [])
        tag_count = len(tags) if isinstance(tags, list) else 0
        score = perf.get("tag_detection_score", tag_count * 0.3)

        return {
            "avg_performance": score,
            "theoretical_fit": score,
            "tags_found": float(tag_count),
            "zoom_layer_handled": 1.0 if state.get("frame_info", {}).get("zoom_level", 1) > 1.5 else 0.0
        }

    def get_iphone_baseline_comparison(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Returns approximate comparison metrics vs modern iPhone (Neural Engine + Vision).
        Use this for internal benchmarking. Real M4 tests will give accurate numbers.
        """
        tags = state.get("detected_tags", [])
        zoom = state.get("frame_info", {}).get("zoom_level", 1.0)

        # Rough iPhone baseline simulation (very strong at this task)
        iphone_accuracy = 0.92 if len(tags) > 0 else 0.75
        our_score = self.evaluate(state).get("theoretical_fit", 0)

        return {
            "our_theoretical_fit": our_score,
            "iphone_estimated_accuracy": iphone_accuracy,
            "relative_efficiency_estimate": our_score / max(iphone_accuracy, 0.01),
            "zoom_layer_advantage": 1.0 if zoom > 2.0 else 0.8
        }


class VoiceVerificationEngine(InferenceEngine):
    def __init__(self, name: str = "voice_verification"):
        super().__init__(name)
        self.theoretical_math_fn: Optional[Callable] = None

    def set_theoretical_math(self, fn: Callable):
        self.theoretical_math_fn = fn

    def train_step(self, state: Dict[str, Any], outcome: float) -> Dict[str, Any]:
        features = state.get("audio_features", {})
        performance = state.get("performance", {})

        if self.theoretical_math_fn is not None:
            try:
                result = self.theoretical_math_fn(features, outcome)
                if isinstance(result, dict):
                    performance.update(result)
            except Exception as e:
                print(f"[VoiceVerificationEngine] Error: {e}")
        else:
            energy = features.get("energy", 0.0)
            performance["voice_score"] = energy * 10

        state["performance"] = performance
        return state

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        perf = state.get("performance", {})
        score = perf.get("voice_score", perf.get("avg_performance", 0))
        return {
            "avg_performance": score,
            "theoretical_fit": score,
            "is_human": 1.0 if score > 0.15 else 0.0
        }


class InferenceEngineComparison:
    def __init__(self):
        self.engines: List[InferenceEngine] = []

    def add_engine(self, engine: InferenceEngine):
        self.engines.append(engine)

    def run_comparison(self, initial_state: Dict[str, Any], num_steps: int = 20, tasks: Optional[List[Dict]] = None) -> Dict[str, Any]:
        results = {}

        for engine in self.engines:
            state = initial_state.copy()
            metrics_over_time = []

            for step in range(num_steps):
                outcome = 0.1
                if tasks and step < len(tasks):
                    outcome = tasks[step].get("outcome", 0.1)

                state = engine.train_step(state, outcome)
                metrics = engine.evaluate(state)
                metrics_over_time.append(metrics)

            final_metrics = metrics_over_time[-1] if metrics_over_time else {}
            results[engine.name] = {
                "final_metrics": final_metrics,
                "history": metrics_over_time,
            }

        results["best_fits"] = self._compute_best_fits(results)
        return results

    def run_benchmark_suite(self, initial_state: Dict[str, Any], trials: int = 5, num_steps: int = 20) -> Dict[str, Any]:
        all_results = []
        summary = {}

        for trial in range(trials):
            result = self.run_comparison(initial_state, num_steps=num_steps)
            all_results.append(result)

        for engine in self.engines:
            name = engine.name
            scores = []
            for res in all_results:
                if name in res:
                    scores.append(res[name]["final_metrics"].get("theoretical_fit", 0))

            if scores:
                summary[name] = {
                    "mean_theoretical_fit": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "best_trial": max(scores),
                }

        best_overall = max(summary, key=lambda k: summary[k]["mean_theoretical_fit"]) if summary else None

        return {
            "summary": summary,
            "best_overall_engine": best_overall,
            "trials_run": trials
        }

    def _compute_best_fits(self, results: Dict[str, Any]) -> Dict[str, str]:
        best = {}
        for metric in ["avg_performance", "theoretical_fit", "is_human", "tags_found"]:
            best_score = -float("inf")
            best_engine = None
            for engine_name, data in results.items():
                if engine_name == "best_fits":
                    continue
                score = data.get("final_metrics", {}).get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_engine = engine_name
            if best_engine:
                best[metric] = best_engine
        return best


if __name__ == "__main__":
    comparator = InferenceEngineComparison()
    vte = VisionTextEngine()
    vte.enable_bitnet_quantization(True)
    comparator.add_engine(vte)

    test_state = {
        "active_profile": "instagram_story_analysis",
        "performance": {},
        "image_features": {"text_density": 0.65, "edge_density": 0.45},
        "frame_info": {"zoom_level": 3.1}
    }

    results = comparator.run_comparison(test_state, num_steps=8)
    print("BitNet VisionTextEngine test:", results)
    print("iPhone baseline comparison:", vte.get_iphone_baseline_comparison(test_state))
