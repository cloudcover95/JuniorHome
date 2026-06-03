# path: src/juniorhome/task_scheduler.py
#!/usr/bin/env python3
"""
JuniorHome Task Scheduler

Simple but production-oriented task scheduler for periodic and one-shot tasks.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        logging.info("TaskScheduler initialized")

    def add_task(self, name: str, func: Callable, interval_seconds: float = 60.0, one_shot: bool = False):
        self.tasks[name] = {
            "func": func,
            "interval": interval_seconds,
            "one_shot": one_shot,
            "last_run": 0.0,
        }
        logging.info(f"Task added: {name} (interval={interval_seconds}s)")

    def _run(self):
        while not self._stop_event.is_set():
            now = time.time()
            for name, task in list(self.tasks.items()):
                if now - task["last_run"] >= task["interval"]:
                    try:
                        task["func"]()
                    except Exception as e:
                        logging.error(f"Task {name} failed: {e}")
                    task["last_run"] = now
                    if task["one_shot"]:
                        del self.tasks[name]
            time.sleep(1)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("TaskScheduler started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        logging.info("TaskScheduler stopped")
