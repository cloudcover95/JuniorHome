# path: src/juniorhome/health_check.py
#!/usr/bin/env python3
"""
Health Check

Production-grade health checking system for JuniorHome.
Monitors critical components (LLM backends, data lake, docker, etc.)
and provides overall system health status.
"""

import logging
import time
from typing import Any, Callable, Dict, List

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class HealthCheck:
    """
    Monitors the health of various system components.
    """

    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        logging.info("HealthCheck initialized")

    def register_check(self, name: str, check_func: Callable[[], bool]):
        self.checks[name] = check_func
        logging.info(f"Health check registered: {name}")

    def run_check(self, name: str) -> Dict[str, Any]:
        if name not in self.checks:
            return {"name": name, "status": "unknown", "error": "Check not found"}

        start = time.time()
        try:
            result = self.checks[name]()
            duration = time.time() - start
            return {
                "name": name,
                "status": "healthy" if result else "unhealthy",
                "duration": duration,
            }
        except Exception as e:
            return {
                "name": name,
                "status": "error",
                "error": str(e),
            }

    def run_all_checks(self) -> Dict[str, Any]:
        results = []
        overall_healthy = True

        for name in self.checks.keys():
            result = self.run_check(name)
            results.append(result)
            if result.get("status") != "healthy":
                overall_healthy = False

        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "checks": results,
            "timestamp": time.time(),
        }

    def get_healthy_components(self) -> List[str]:
        healthy = []
        for name, check in self.checks.items():
            try:
                if check():
                    healthy.append(name)
            except Exception:
                pass
        return healthy
