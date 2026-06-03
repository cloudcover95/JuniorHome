# path: src/juniorhome/resilient_knowledge_pipeline.py
#!/usr/bin/env python3
"""
Resilient Knowledge Pipeline

Production-grade version of the knowledge pipeline with:
- Automatic error handling and fallback routing (Ollama ↔ BitNet-mlx)
- File change triggers with debouncing
- Better resilience for long-running monitoring

This is designed for real-world, always-on use in the ecosystem.
"""

import logging
import time
from pathlib import Path
from threading import Timer
from typing import Any, Callable, Dict, List, Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

from .smart_llm_router import SmartLLMRouter
from .ternary_integration import TernaryIntegration

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ResilientKnowledgePipeline:
    """
    Robust knowledge processing pipeline with error handling and file triggers.
    """

    def __init__(
        self,
        vault_path: str,
        use_ternary: bool = False,
        debounce_seconds: float = 2.0,
        on_important_finding: Optional[Callable[[str, str], None]] = None,
    ):
        self.vault_path = Path(vault_path)
        self.use_ternary = use_ternary
        self.debounce_seconds = debounce_seconds
        self.on_important_finding = on_important_finding

        self.llm_router = SmartLLMRouter()
        self.ternary = TernaryIntegration() if use_ternary else None

        self._pending_files: Dict[str, Timer] = {}
        self.processed_files: set = set()

        if not self.vault_path.exists():
            self.vault_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"ResilientKnowledgePipeline watching: {self.vault_path}")

    def _safe_route(self, prompt: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        try:
            return self.llm_router.route(prompt, prefer_bitnet=prefer_bitnet)
        except Exception as e:
            logging.warning(f"Primary routing failed: {e}. Trying fallback...")
            # Try opposite backend as fallback
            try:
                return self.llm_router.route(prompt, prefer_bitnet=not prefer_bitnet)
            except Exception as e2:
                logging.error(f"Both backends failed: {e2}")
                return {"error": str(e2), "backend": "none"}

    def _process_file_safely(self, file_path: Path):
        try:
            if not file_path.is_file() or file_path.suffix.lower() not in [".md", ".txt"]:
                return

            content = file_path.read_text(encoding="utf-8")

            prompt = f"""Analyze this note. Determine importance and whether it should be expanded.

File: {file_path.name}

{content[:2500]}

Respond with: Importance (High/Medium/Low) and one-sentence summary."""

            result = self._safe_route(prompt, prefer_bitnet=self.use_ternary)

            if "High" in result.get("response", "") and self.on_important_finding:
                self.on_important_finding(str(file_path), result.get("response", ""))

            self.processed_files.add(str(file_path))

        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    def _debounced_process(self, file_path: Path):
        key = str(file_path)

        if key in self._pending_files:
            self._pending_files[key].cancel()

        timer = Timer(self.debounce_seconds, lambda: self._process_file_safely(file_path))
        self._pending_files[key] = timer
        timer.start()

    def start_watching(self):
        if not HAS_WATCHDOG:
            logging.warning("watchdog not available. Using polling.")
            self._run_polling()
            return

        class Handler(FileSystemEventHandler):
            def __init__(self, processor):
                self.processor = processor

            def on_created(self, event):
                if not event.is_directory:
                    self.processor._debounced_process(Path(event.src_path))

            def on_modified(self, event):
                if not event.is_directory:
                    self.processor._debounced_process(Path(event.src_path))

        observer = Observer()
        observer.schedule(Handler(self), str(self.vault_path), recursive=True)
        observer.start()

        logging.info("Resilient file watching started")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def _run_polling(self):
        logging.info("Running in polling mode")
        while True:
            for file_path in self.vault_path.rglob("*"):
                if file_path.is_file():
                    self._debounced_process(file_path)
            time.sleep(30)
