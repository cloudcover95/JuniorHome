# src/physiomanifold/ui_terraforming/agent.py

import mlx.core as mx
from mlx_lm import load, generate
import logging

class TerraformingAgent:
    """
    The Local AGI Frontend Developer (JuniorHome OS).
    Translates physical states or user requests into executable HTML/JS components.
    """
    def __init__(self, model_path="mlx-community/Meta-Llama-3-8B-Instruct-4bit"):
        self.model_path = model_path
        self.model, self.tokenizer = None, None
        self._load_model()

    def _load_model(self):
        try:
            self.model, self.tokenizer = load(self.model_path)
            logging.info("JuniorHome Terraforming Agent loaded to MLX Unified Memory.")
        except Exception as e:
            logging.error(f"Node Failure: LLM Matrix allocation failed. Error: {e}")

    def generate_board(self, system_context: str, component_request: str) -> str:
        if not self.model:
            return "<div class='text-red-500'>AGENT OFFLINE</div>"

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the JuniorHome OS UI Harvester. Output ONLY raw, executable HTML/TailwindCSS/JS.
Do not use markdown backticks. Do not apologize.
System Context: {system_context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Build component: {component_request}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
        return response.replace("```html", "").replace("```", "").strip()
