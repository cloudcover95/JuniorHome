# path: src/juniorhome/tri_state_execution_engine.py
#!/usr/bin/env python3
"""
TriStateExecutionEngine

Clean, isolated implementation of the three black boxes defined in the
JUNIOR_OS_DIRECTIVE.

This module enforces strict separation while allowing the SovereignEdgeOrchestrator
to route intelligently.

All three boxes can eventually delegate to the JuniorOS kernel when available.
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
    """
    User Black Box (Original Inference)
    Strict 1.58-bit ternary + manifold folding. Maximum sovereignty.
    """

    def execute(self, state: Any) -> Dict[str, Any]:
        if not HAS_MANIFOLD:
            return {"error": "ManifoldFoldingQuantizer not available"}

        result = fold_manifold_full(state)
        result["box"] = "user"
        result["sovereignty_level"] = "maximum"
        return result


class SwarmBlackBox:
    """
    Swarm Black Box (Agent Orchestrator / 2nd Brain)
    Uses full agent debate + TDA + Second Brain cross-pollination.
    """

    def __init__(self, agent_orchestrator):
        self.agent_orchestrator = agent_orchestrator

    def execute(self, state: Any, agent_context: Optional[Any] = None) -> Dict[str, Any]:
        # Use the full AgentOrchestrator capabilities
        routed = self.agent_orchestrator.route_intelligence(
            state, mode="swarm", agent_context=agent_context
        )

        # Also run manifold analysis for TDA insights
        if HAS_MANIFOLD:
            manifold_result = fold_manifold_full(state)
            routed["manifold_insights"] = manifold_result.get("tda", {})
            routed["persistence_signature"] = manifold_result.get("persistence_signature", {})

        routed["box"] = "swarm"
        return routed


class IndustryFallbackBox:
    """
    Industry Fallback Black Box
    Dense matrix path used when ternary coherence is insufficient.
    """

    def execute(self, state: Any) -> Dict[str, Any]:
        if HAS_TRISTATE:
            router = TriStateRouter()
            result = router.route_industry_fallback(state)
        else:
            # Simple dense fallback
            import numpy as np
            weights = np.random.standard_normal(state.shape)
            result = np.dot(state, weights.T)

        return {
            "result": result,
            "box": "industry",
            "note": "Dense fallback path activated",
        }


class TriStateExecutionEngine:
    """
    Central engine that manages the three black boxes.
    Provides a clean interface for the SovereignEdgeOrchestrator.
    """

    def __init__(self, agent_orchestrator):
        self.user_box = UserBlackBox()
        self.swarm_box = SwarmBlackBox(agent_orchestrator)
        self.industry_box = IndustryFallbackBox()
        self.router = TriStateRouter() if HAS_TRISTATE else None
        logging.info("TriStateExecutionEngine initialized")

    def execute(self, state: Any, mode: str = "auto", agent_context: Any = None) -> Dict[str, Any]:
        if mode == "user":
            return self.user_box.execute(state)
        elif mode == "swarm":
            return self.swarm_box.execute(state, agent_context=agent_context)
        elif mode == "industry":
            return self.industry_box.execute(state)

        # Auto mode with coherence-based routing
        if self.router:
            return self.router.evaluate_and_route(state, agent_context=agent_context, mode="auto")

        # Default to user box if no router
        return self.user_box.execute(state)

    def get_box_status(self) -> Dict[str, str]:
        return {
            "user_black_box": "active",
            "swarm_black_box": "active",
            "industry_fallback": "active",
        }
