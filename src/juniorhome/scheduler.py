# path: src/juniorhome/scheduler.py
#!/usr/bin/env python3
"""
Scheduler

Simple scheduler for running workflows periodically (e.g. nightly processing).
Supports cron-like scheduling for automated tasks like Obsidian data processing.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Scheduler:
    """
    Lightweight scheduler for periodic task execution.
    """

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        logging.info("Scheduler initialized")

    def add_daily_task(self, name: str, func: Callable, hour: int = 2, minute: int = 0):
        self.tasks[name] = {
            "func": func,
            "type": "daily",
            "hour": hour,
            "minute": minute,
            "last_run": None,
        }
        logging.info(f"Daily task added: {name} at {hour:02d}:{minute:02d}")

    def add_interval_task(self, name: str, func: Callable, interval_seconds: int):
        self.tasks[name] = {
            "func": func,
            "type": "interval",
            "interval": interval_seconds,
            "last_run": 0,
        }
        logging.info(f"Interval task added: {name} every {interval_seconds}s")

    def _should_run_daily(self, task: Dict) -> bool:
        now = datetime.now()
        if task["last_run"] is None:
            return now.hour == task["hour"] and now.minute >= task["minute"]

        last = task["last_run"]
        if last.date() < now.date():
            return now.hour == task["hour"] and now.minute >= task["minute"]
        return False

    def _run(self):
        while not self._stop_event.is_set():
            now = time.time()
            for name, task in list(self.tasks.items()):
                should_run = False

                if task["type"] == "daily":
                    should_run = self._should_run_daily(task)
                elif task["type"] == "interval":
                    if now - task.get("last_run", 0) >= task["interval"]:
                        should_run = True

                if should_run:
                    try:
                        logging.info(f"Running scheduled task: {name}")
                        task["func"]()
                        task["last_run"] = now if task["type"] == "interval" else datetime.now()
                    except Exception as e:
                        logging.error(f"Scheduled task {name} failed: {e}")

            time.sleep(30)  # Check every 30 seconds

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("Scheduler started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logging.info("Scheduler stopped")
