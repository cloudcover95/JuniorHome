# path: src/juniorhome/observability_manager.py
#!/usr/bin/env python3
"""
Observability Manager

Central manager that combines HealthCheck, MetricsCollector,
TracingContext, and EventBus into one unified observability layer.
"""

import logging
from typing import Any, Dict, Optional

from .health_check import HealthCheck
from .metrics_collector import MetricsCollector
from .tracing_context import TracingContext
from .event_bus import EventBus

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ObservabilityManager:
    """
    Unified observability layer for JuniorHome.
    """

    def __init__(self):
        self.health = HealthCheck()
        self.metrics = MetricsCollector()
        self.tracing = TracingContext()
        self.event_bus = EventBus()
        logging.info("ObservabilityManager initialized")

    def record_metric(self, name: str, value: float, metric_type: str = "gauge"):
        if metric_type == "counter":
            self.metrics.increment(name, int(value))
        elif metric_type == "gauge":
            self.metrics.set_gauge(name, value)
        elif metric_type == "timer":
            self.metrics.record_timer(name, value)

    def start_operation(self, name: str) -> str:
        return self.tracing.start_trace(name)

    def end_operation(self, trace_id: Optional[str] = None):
        self.tracing.end_trace(trace_id)

    def publish_event(self, topic: str, data: Any = None):
        self.event_bus.publish(topic, data)

    def subscribe_to_event(self, topic: str, callback: Any):
        self.event_bus.subscribe(topic, callback)

    def run_health_checks(self) -> Dict[str, Any]:
        return self.health.run_all_checks()

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "health": self.health.run_all_checks(),
            "metrics": self.metrics.get_all_metrics(),
            "current_trace": self.tracing.get_current_trace(),
        }
