# path: src/juniorhome/animated_visualization.py
#!/usr/bin/env python3
"""
Animated Visualization

Extends visualization capabilities with dynamic, LLM-controlled animations.
Supports real-time parameter updates for Three.js scenes via the SmartLLMRouter.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .smart_llm_router import SmartLLMRouter
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False
    SmartLLMRouter = None

from .visualization_manager import VisualizationManager

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AnimatedVisualization:
    """
    Manages animated Three.js visualizations that can be controlled
    in real-time by local LLMs.
    """

    def __init__(self, visualization_manager: Optional[VisualizationManager] = None, llm_router: Optional[Any] = None):
        self.viz_manager = visualization_manager or VisualizationManager()

        if HAS_ROUTER and llm_router is None:
            self.llm_router = SmartLLMRouter()
        else:
            self.llm_router = llm_router

        logging.info("AnimatedVisualization initialized")

    def create_animated_manifold(self, name: str, points: List[List[float]], animation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Creates a Three.js scene with basic animation that can be updated via LLM.
        """
        params = animation_params or {
            "rotation_speed": 0.01,
            "color": "#00ff00",
            "scale": 1.0
        }

        data = {
            "points": points,
            "type": "animated_manifold",
            "animation": params
        }

        return self.viz_manager.create_threejs_scene(name, data)

    def update_animation_params(self, name: str, new_params: Dict[str, Any]) -> bool:
        """
        Updates animation parameters of an existing visualization.
        Can be called by LLMs for real-time control.
        """
        return self.viz_manager.update_visualization(name, new_params)

    def llm_controlled_update(self, name: str, instruction: str) -> Dict[str, Any]:
        """
        Uses the LLM router to interpret natural language instructions
        and update the visualization accordingly.
        """
        if not self.llm_router:
            return {"error": "No LLM router available"}

        prompt = f"""You control a 3D visualization called '{name}'.
Current instruction: {instruction}

Respond ONLY with a JSON object containing animation parameters to change, for example:
{{
  "rotation_speed": 0.05,
  "color": "#ff0000",
  "scale": 1.5
}}
"""

        result = self.llm_router.route(prompt, prefer_bitnet=False)
        response_text = result.get("response", "")

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                success = self.update_animation_params(name, params)
                return {
                    "success": success,
                    "applied_params": params,
                    "raw_response": response_text
                }
        except Exception as e:
            return {
                "error": f"Failed to parse LLM response: {e}",
                "raw_response": response_text
            }

        return {"error": "Could not extract valid parameters", "raw_response": response_text}
