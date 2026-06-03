# path: src/juniorhome/tracing_context.py
#!/usr/bin/env python3
"""
Tracing Context

Provides request/operation tracing across JuniorHome components.
Works with EventBus and ActionAuditor for full observability.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TracingContext:
    """
    Manages tracing context for operations across the system.
    """

    def __init__(self):
        self.current_trace_id: Optional[str] = None
        self.spans: Dict[str, Dict[str, Any]] = {}
        logging.info("TracingContext initialized")

    def start_trace(self, operation: str) -> str:
        trace_id = str(uuid.uuid4())
        self.current_trace_id = trace_id
        self.spans[trace_id] = {
            "operation": operation,
            "start_time": time.time(),
            "events": [],
        }
        logging.debug(f"Started trace {trace_id} for {operation}")
        return trace_id

    def end_trace(self, trace_id: Optional[str] = None):
        trace_id = trace_id or self.current_trace_id
        if trace_id in self.spans:
            self.spans[trace_id]["end_time"] = time.time()
            duration = self.spans[trace_id]["end_time"] - self.spans[trace_id]["start_time"]
            logging.debug(f"Ended trace {trace_id} (duration: {duration:.3f}s)")

    def add_event(self, event: str, data: Optional[Dict[str, Any]] = None):
        if self.current_trace_id and self.current_trace_id in self.spans:
            self.spans[self.current_trace_id]["events"].append({
                "timestamp": time.time(),
                "event": event,
                "data": data or {},
            })

    @contextmanager
    def trace(self, operation: str):
        trace_id = self.start_trace(operation)
        try:
            yield trace_id
        finally:
            self.end_trace(trace_id)

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return self.spans.get(trace_id)

    def get_current_trace(self) -> Optional[Dict[str, Any]]:
        if self.current_trace_id:
            return self.spans.get(self.current_trace_id)
        return None
