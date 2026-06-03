# path: src/juniorhome/metrics_collector.py
#!/usr/bin/env python3
"""
Metrics Collector

Simple but production-useful metrics collection system.
Supports counters, gauges, and basic timing.
Can be extended later for Prometheus, StatsD, or custom exporters.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class MetricsCollector:
    """
    Lightweight metrics collection.
    """

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        logging.info("MetricsCollector initialized")

    def increment(self, name: str, value: int = 1):
        self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        self.gauges[name] = value

    def record_timer(self, name: str, duration: float):
        self.timers[name].append(duration)

    def time_it(self, name: str):
        """Context manager / decorator for timing operations."""
        class _Timer:
            def __init__(self, collector, metric_name):
                self.collector = collector
                self.name = metric_name
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                duration = time.time() - self.start
                self.collector.record_timer(self.name, duration)

        return _Timer(self, name)

    def get_all_metrics(self) -> Dict[str, Any]:
        metrics = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timers": {k: {
                "count": len(v),
                "avg": sum(v) / len(v) if v else 0,
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
            } for k, v in self.timers.items()},
        }
        return metrics

    def reset(self):
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()
        logging.info("Metrics reset")
