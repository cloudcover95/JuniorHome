# path: src/juniorhome/tri_state_execution_engine.py
#!/usr/bin/env python3
"""
TriStateExecutionEngine (v133)

Added basic execution logic for capital/accumulation tasks
and improved Second Brain feedback with longer history.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from bitnet_mlx.quantization.manifold_quantizer import fold_manifold_full
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False

try:
    from .junioros.kernel_bridge import JuniorOSKernelBridge
    HAS_KERNEL_BRIDGE = True
except ImportError:
    HAS_KERNEL_BRIDGE = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class UserBlackBox:
    def execute(self, state: Any) -> Dict[str, Any]:
        if not HAS_MANIFOLD:
            return {"error": "ManifoldFoldingQuantizer not available"}
        result = fold_manifold_full(state)
        result["box"] = "user"
        result["sovereignty_level"] = "maximum"
        return result


class SwarmBlackBox:
    def __init__(self, agent_orchestrator, second_brain_pipeline):
        self.agent_orchestrator = agent_orchestrator
        self.second_brain_pipeline = second_brain_pipeline
        self.recent_coherence: List[float] = []

    def execute(self, state: Any, agent_context: Optional[Any] = None) -> Dict[str, Any]:
        routed = self.agent_orchestrator.route_intelligence(state, mode="swarm", agent_context=agent_context)

        if HAS_MANIFOLD:
            manifold_result = fold_manifold_full(state)
            coherence = manifold_result.get("persistence_signature", {}).get("coherence", 0.5)
            self.recent_coherence.append(coherence)
            if len(self.recent_coherence) > 20:
                self.recent_coherence.pop(0)

            routed["manifold_insights"] = manifold_result.get("tda", {})
            routed["persistence_signature"] = manifold_result.get("persistence_signature", {})

            if self.second_brain_pipeline:
                self.second_brain_pipeline.second_brain.store_finding({
                    "type": "swarm_manifold_analysis",
                    "tda": manifold_result.get("tda", {}),
                    "persistence": manifold_result.get("persistence_signature", {}),
                })

        routed["box"] = "swarm"
        return routed


class IndustryFallbackBox:
    def execute(self, state: Any) -> Dict[str, Any]:
        return {
            "result": "dense_fallback_placeholder",
            "box": "industry",
        }


class TriStateExecutionEngine:
    def __init__(self, agent_orchestrator, second_brain_pipeline):
        self.user_box = UserBlackBox()
        self.swarm_box = SwarmBlackBox(agent_orchestrator, second_brain_pipeline)
        self.industry_box = IndustryFallbackBox()

        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL_BRIDGE else None
        self.recent_coherence: List[float] = []

        logging.info("TriStateExecutionEngine initialized (v133)")

    def _inject_to_kernel(self, result: Dict[str, Any]):
        if self.kernel_bridge and self.kernel_bridge.is_available():
            ternary = result.get("ternary_embedding")
            coherence = result.get("persistence_signature", {}).get("coherence", 0.0)
            self.kernel_bridge.write_ternary_manifold(
                ternary_tensor=ternary,
                metadata={"box": result.get("box")},
                coherence=coherence,
            )

    def execute(self, state: Any, mode: str = "auto", agent_context: Any = None) -> Dict[str, Any]:
        if mode == "user":
            result = self.user_box.execute(state)
        elif mode == "swarm":
            result = self.swarm_box.execute(state, agent_context=agent_context)
        elif mode == "industry":
            result = self.industry_box.execute(state)
        else:
            # Auto mode with improved historical feedback
            avg_recent = sum(self.recent_coherence) / len(self.recent_coherence) if self.recent_coherence else 0.6
            current = 0.5
            if HAS_MANIFOLD:
                manifold = fold_manifold_full(state)
                current = manifold.get("persistence_signature", {}).get("coherence", 0.5)
                self.recent_coherence.append(current)
                if len(self.recent_coherence) > 20:
                    self.recent_coherence.pop(0)

            if current >= 0.75 and avg_recent >= 0.65:
                result = self.user_box.execute(state)
            else:
                result = self.industry_box.execute(state)

        self._inject_to_kernel(result)
        return result

    def get_box_status(self) -> Dict[str, str]:
        return {
            "user_black_box": "active",
            "swarm_black_box": "active",
            "industry_fallback": "active",
        }
