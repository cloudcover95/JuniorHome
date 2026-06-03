# path: src/juniorhome/smart_llm_router.py
#!/usr/bin/env python3
"""
Smart LLM Router

Intelligent routing between local Ollama (fast daily work)
and BitNet-mlx ternary pipeline (math-heavy / deterministic tasks).

End-user friendly with simple fallback logic.
"""

import logging
from typing import Any, Dict, Optional

from .ollama_integration import OllamaIntegration
from .ternary_integration import TernaryIntegration

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SmartLLMRouter:
    """
    Routes requests between Ollama and BitNet-mlx based on task type.
    """

    def __init__(self):
        self.ollama = OllamaIntegration()
        self.bitnet = TernaryIntegration()

        self.ollama_available = self.ollama.available
        self.bitnet_available = self.bitnet.is_available() if self.bitnet else False

        logging.info(f"SmartLLMRouter ready | Ollama: {self.ollama_available} | BitNet: {self.bitnet_available}")

    def route(
        self,
        prompt: str,
        prefer_bitnet: bool = False,
        model: str = "llama3.2",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route request to the best available backend.
        """
        # Force BitNet path if requested and available
        if prefer_bitnet and self.bitnet_available:
            return self._run_bitnet(prompt, metadata)

        # Default: try Ollama first for speed
        if self.ollama_available:
            result = self.ollama.generate(model=model, prompt=prompt)
            if "error" not in result:
                result["backend"] = "ollama"
                return result

        # Fallback to BitNet if Ollama fails or unavailable
        if self.bitnet_available:
            return self._run_bitnet(prompt, metadata)

        return {"error": "No LLM backend available"}

    def _run_bitnet(self, prompt: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = self.bitnet.analyze(prompt, metadata=metadata)
        result["backend"] = "bitnet-mlx"
        return result

    def status(self) -> Dict[str, Any]:
        return {
            "ollama_available": self.ollama_available,
            "bitnet_available": self.bitnet_available,
            "recommended_for_daily": "ollama" if self.ollama_available else "bitnet-mlx",
        }
