# path: src/juniorhome/tri_state_execution_engine.py
#!/usr/bin/env python3
"""
TriStateExecutionEngine (with Second Brain Feedback)

Enhanced version that uses TDA / persistence signatures from the
ManifoldFoldingQuantizer to make smarter routing decisions.

Also includes basic feedback loop into the SecondBrainPipeline.
"""

import logging
from typing import Any, Dict, Optional

try:
    from bitnet_mlx.quantization.manifold_quantizer import fold_manifold_full
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False

try:
    from bitnet_mlx.compute.tri_state_router import TriStateRouter
    HAS_TRISTATE = True
except ImportError:
    HAS_TRISTATE = False

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

    def execute(self, state: Any, agent_context: Optional[Any] = None) -> Dict[str, Any]:
        routed = self.agent_orchestrator.route_intelligence(
            state, mode="swarm", agent_context=agent_context
        )

        if HAS_MANIFOLD:
            manifold_result = fold_manifold_full(state)
            routed["manifold_insights"] = manifold_result.get("tda", {})
            routed["persistence_signature"] = manifold_result.get("persistence_signature", {})

            # Feedback into Second Brain
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
        if HAS_TRISTATE:
            router = TriStateRouter()
            result = router.route_industry_fallback(state)
        else:
            import numpy as np
            weights = np.random.standard_normal(state.shape)
            result = np.dot(state, weights.T)

        return {
            "result": result,
            "box": "industry",
            "note": "Dense fallback path activated",
        }


class TriStateExecutionEngine:
    def __init__(self, agent_orchestrator, second_brain_pipeline):
        self.user_box = UserBlackBox()
        self.swarm_box = SwarmBlackBox(agent_orchestrator, second_brain_pipeline)
        self.industry_box = IndustryFallbackBox()
        self.router = TriStateRouter() if HAS_TRISTATE else None
        logging.info("TriStateExecutionEngine initialized with feedback loop")

    def execute(self, state: Any, mode: str = "auto", agent_context: Any = None) -> Dict[str, Any]:
        if mode == "user":
            return self.user_box.execute(state)
        elif mode == "swarm":
            return self.swarm_box.execute(state, agent_context=agent_context)
        elif mode == "industry":
            return self.industry_box.execute(state)

        # Auto mode with improved coherence detection
        if self.router:
            # Use manifold analysis to help decide
            if HAS_MANIFOLD:
                manifold = fold_manifold_full(state)
                coherence = manifold.get("persistence_signature", {}).get("coherence", 0.5)

                if coherence >= self.router.tda_threshold:
                    return self.user_box.execute(state)
                else:
                    return self.industry_box.execute(state)

            return self.router.evaluate_and_route(state, agent_context=agent_context, mode="auto")

        return self.user_box.execute(state)

    def get_box_status(self) -> Dict[str, str]:
        return {
            "user_black_box": "active",
            "swarm_black_box": "active",
            "industry_fallback": "active",
        }
