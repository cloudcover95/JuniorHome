# path: src/juniorhome/ollama_integration.py
#!/usr/bin/env python3
"""
Ollama Integration

Clean integration layer for running local LLMs via Ollama.
Designed for immediate daily productivity (70-80% of workflows).
Works alongside BitNet-mlx ternary pipeline for math-heavy tasks.
"""

import logging
import requests
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class OllamaIntegration:
    """
    Simple, production-ready client for Ollama running on localhost.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.available = self._check_connection()

        if self.available:
            logging.info(f"Ollama connected at {self.base_url}")
        else:
            logging.warning("Ollama not available. Start with: ollama serve")

    def _check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        if not self.available:
            return []
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        if not self.available:
            return {"error": "Ollama not available"}

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            data = response.json()
            return {
                "response": data.get("response", ""),
                "model": model,
                "done": data.get("done", True),
            }
        except Exception as e:
            return {"error": str(e)}

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        if not self.available:
            return {"error": "Ollama not available"}

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            data = response.json()
            return {
                "message": data.get("message", {}),
                "model": model,
            }
        except Exception as e:
            return {"error": str(e)}

    def pull_model(self, model_name: str) -> bool:
        if not self.available:
            return False
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600,
            )
            return response.status_code == 200
        except Exception:
            return False
