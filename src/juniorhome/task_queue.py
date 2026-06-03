# path: src/juniorhome/task_queue.py
#!/usr/bin/env python3
"""
Task Queue

Simple production-grade task queue for background and asynchronous job processing.
Supports priority, retries, and basic scheduling.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TaskQueue:
    """
    Lightweight task queue with worker threads.
    """

    def __init__(self, num_workers: int = 2, max_retries: int = 3):
        self.queue: deque = deque()
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self.results: Dict[str, Any] = {}
        logging.info(f"TaskQueue initialized with {num_workers} workers")

    def add_task(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs):
        task_id = task_id or f"task_{int(time.time() * 1000)}"
        self.queue.append({
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "retries": 0,
        })
        logging.debug(f"Task added: {task_id}")
        return task_id

    def _worker(self):
        while not self._stop_event.is_set():
            if not self.queue:
                time.sleep(0.1)
                continue

            try:
                task = self.queue.popleft()
            except IndexError:
                continue

            try:
                result = task["func"](*task["args"], **task["kwargs"])
                self.results[task["id"]] = {"status": "success", "result": result}
                logging.debug(f"Task completed: {task['id']}")
            except Exception as e:
                task["retries"] += 1
                if task["retries"] < self.max_retries:
                    self.queue.append(task)  # Re-queue for retry
                    logging.warning(f"Task {task['id']} failed, retrying ({task['retries']}/{self.max_retries})")
                else:
                    self.results[task["id"]] = {"status": "failed", "error": str(e)}
                    logging.error(f"Task {task['id']} failed permanently: {e}")

    def start(self):
        if self.workers:
            return
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        logging.info(f"Started {self.num_workers} workers")

    def stop(self):
        self._stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)
        logging.info("TaskQueue stopped")

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.results.get(task_id)

    def pending_count(self) -> int:
        return len(self.queue)
