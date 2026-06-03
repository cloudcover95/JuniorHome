# path: src/juniorhome/health_monitor.py
#!/usr/bin/env python3
"""
JuniorHome Health Monitor

Basic health monitoring for components in the orchestrator.
"""

import logging
import time
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class HealthMonitor:
    def __init__(self):
        self.components: Dict[str, Dict[str, Any]] = {}
        logging.info("HealthMonitor initialized")

    def register_component(self, name: str, check_func: Any = None):
        self.components[name] = {
            "status": "unknown",
            "last_check": 0.0,
            "check_func": check_func,
        }

    def check_all(self) -> Dict[str, str]:
        results = {}
        for name, comp in self.components.items():
            if comp["check_func"]:
                try:
                    status = "healthy" if comp["check_func"]() else "unhealthy"
                except Exception:
                    status = "error"
            else:
                status = "unknown"
            comp["status"] = status
            comp["last_check"] = time.time()
            results[name] = status
        return results

    def get_status(self) -> Dict[str, Any]:
        return {name: comp["status"] for name, comp in self.components.items()}
