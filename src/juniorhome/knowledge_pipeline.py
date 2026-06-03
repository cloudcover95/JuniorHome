# path: src/juniorhome/knowledge_pipeline.py
#!/usr/bin/env python3
"""
Knowledge Pipeline

Unified pipeline that combines:
- Obsidian vault monitoring
- Data stream processing (NotebookLM, audio, bookmarks, etc.)
- Assessment via SmartLLMRouter or Ternary pipeline
- Structured storage and optional expansion

This is a core component for automated knowledge systems in JuniorHome.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .obsidian_vault_processor import ObsidianVaultProcessor
from .smart_llm_router import SmartLLMRouter
from .ternary_integration import TernaryIntegration

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class KnowledgePipeline:
    """
    End-to-end knowledge processing pipeline.
    Monitors data streams, assesses importance, and routes to appropriate backends.
    """

    def __init__(
        self,
        vault_path: str,
        use_ternary: bool = False,
        on_important_finding: Optional[Callable[[str, str], None]] = None,
    ):
        self.vault_path = Path(vault_path)
        self.use_ternary = use_ternary
        self.on_important_finding = on_important_finding

        self.llm_router = SmartLLMRouter()
        self.ternary = TernaryIntegration() if use_ternary else None

        self.vault_processor = ObsidianVaultProcessor(
            vault_path=str(self.vault_path),
            llm_router=self.llm_router,
            on_important_finding=self._handle_important_finding,
        )

        logging.info(f"KnowledgePipeline initialized (ternary={use_ternary})")

    def _handle_important_finding(self, file_path: str, assessment: str):
        logging.info(f"Important finding detected in {file_path}")
        if self.on_important_finding:
            self.on_important_finding(file_path, assessment)

    def process_once(self) -> List[Dict[str, Any]]:
        return self.vault_processor.scan_once()

    def start_monitoring(self):
        logging.info("Starting continuous monitoring of Obsidian vault...")
        self.vault_processor.start_watching()

    def analyze_with_best_backend(self, text: str, prefer_ternary: bool = False) -> Dict[str, Any]:
        if prefer_ternary and self.ternary and self.ternary.is_available():
            return self.ternary.analyze(text)
        return self.llm_router.route(text, prefer_bitnet=prefer_ternary)

    def expand_on_finding(self, content: str) -> Dict[str, Any]:
        prompt = f"""Take this finding and expand on it with deeper analysis, implications, and next steps:

{content}
"""
        return self.llm_router.route(prompt, prefer_bitnet=False)
