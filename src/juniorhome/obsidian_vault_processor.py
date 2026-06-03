# path: src/juniorhome/obsidian_vault_processor.py
#!/usr/bin/env python3
"""
Obsidian Vault Processor

Monitors an Obsidian folder for new data streams
(NotebookLM, audio transcripts, X bookmarks, etc.),
parses them, runs assessments, and organizes important findings.

Designed to feed into JuniorMemSys and expand on key information.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

from .smart_llm_router import SmartLLMRouter

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ObsidianVaultProcessor:
    """
    Watches an Obsidian vault folder and processes new files.
    """

    def __init__(
        self,
        vault_path: str,
        llm_router: Optional[SmartLLMRouter] = None,
        on_important_finding: Optional[Callable[[str, str], None]] = None,
    ):
        self.vault_path = Path(vault_path)
        self.llm_router = llm_router or SmartLLMRouter()
        self.on_important_finding = on_important_finding
        self.processed_files: set = set()

        if not self.vault_path.exists():
            self.vault_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"ObsidianVaultProcessor watching: {self.vault_path}")

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.is_file() or file_path.suffix.lower() not in [".md", ".txt"]:
            return {"skipped": True}

        if str(file_path) in self.processed_files:
            return {"already_processed": True}

        try:
            content = file_path.read_text(encoding="utf-8")

            # Simple assessment prompt
            prompt = f"""Analyze this note and determine if it contains important findings worth expanding on.

File: {file_path.name}

Content:
{content[:3000]}

Respond with:
- Importance: High / Medium / Low
- Summary: One sentence summary
- Should expand: Yes / No
"""

            result = self.llm_router.route(prompt, prefer_bitnet=False)

            assessment = {
                "file": str(file_path),
                "processed_at": time.time(),
                "llm_response": result.get("response", ""),
            }

            self.processed_files.add(str(file_path))

            # Trigger callback for important findings
            if self.on_important_finding and "High" in result.get("response", ""):
                self.on_important_finding(str(file_path), result.get("response", ""))

            return assessment

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return {"error": str(e)}

    def scan_once(self) -> List[Dict[str, Any]]:
        results = []
        for file_path in self.vault_path.rglob("*"):
            if file_path.is_file():
                result = self.process_file(file_path)
                if not result.get("skipped") and not result.get("already_processed"):
                    results.append(result)
        return results

    def start_watching(self):
        if not HAS_WATCHDOG:
            logging.warning("watchdog not installed. Falling back to polling mode.")
            self._poll_mode()
            return

        class Handler(FileSystemEventHandler):
            def __init__(self, processor):
                self.processor = processor

            def on_created(self, event):
                if not event.is_directory:
                    self.processor.process_file(Path(event.src_path))

        observer = Observer()
        observer.schedule(Handler(self), str(self.vault_path), recursive=True)
        observer.start()
        logging.info(f"Started watching {self.vault_path}")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def _poll_mode(self):
        logging.info("Running in polling mode (watchdog not available)")
        while True:
            self.scan_once()
            time.sleep(60)  # Check every minute
