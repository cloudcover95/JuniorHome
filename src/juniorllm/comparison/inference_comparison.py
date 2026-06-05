# path: src/juniorllm/comparison/inference_comparison.py

"""
InferenceEngineComparison

Enhanced with BitNetVoiceEngine and a proper benchmark suite
for running real theoretical math functions and finding best fits.

Supports:
- Multiple engines (baseline, theoretical math variants, BitNet voice path)
- Multi-trial benchmarking with stats
- Direct use with DigitalCallManager for voice verification
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
    """
    Specialized engine using BitNet (quantized) inference for voice verification.

    Can be connected to real BitNet-MLX models or the theoretical math pipeline.
    Designed for low-power, sovereign voice recognition in DigitalCallManager.
    """

    def __init__(self, name: str = "bitnet_voice"):
        super().__init__(name)
        self.model_fn: Optional[Callable] = None

    def set_model(self, fn: Callable):
        """Plug in a real BitNet or quantized model for voice scoring."""
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
            # Fallback: simple quantized-like scoring
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
        """
        Run multiple trials and compute statistics for robust best-fit analysis.
        Ideal for comparing real theoretical math functions.
        """
        all_results = []
        summary = {}

        for trial in range(trials):
            result = self.run_comparison(initial_state, num_steps=num_steps)
            all_results.append(result)

        # Aggregate stats per engine
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
        for metric in ["avg_performance", "theoretical_fit", "is_human"]:
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


# Quick example for running benchmarks with real theoretical math
if __name__ == "__main__":
    comparator = InferenceEngineComparison()
    comparator.add_engine(BaselinePlasticityEngine())
    comparator.add_engine(BitNetVoiceEngine())

    # Add your real theoretical math engines here
    # theo1 = TheoreticalMathEngine(name="your_math_v1")
    # theo1.set_theoretical_math(your_real_function)
    # comparator.add_engine(theo1)

    initial = {"active_profile": "voice_verification", "performance": {}, "audio_features": {"energy": 0.25}}
    suite_result = comparator.run_benchmark_suite(initial, trials=3, num_steps=15)
    print(suite_result)
