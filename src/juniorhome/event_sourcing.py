# path: src/juniorhome/event_sourcing.py
#!/usr/bin/env python3
"""
Event Sourcing Foundation

Basic Event Store and Event Sourced Aggregate.
Foundation for implementing Event Sourcing + CQRS patterns.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Event:
    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[float] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class EventStore:
    """
    Simple file-based Event Store.
    In production this would use a proper database or event store.
    """

    def __init__(self, store_path: str = "events.jsonl"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event):
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def get_events(self, limit: Optional[int] = None) -> List[Event]:
        if not self.store_path.exists():
            return []

        events = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(Event(data["event_type"], data["data"], data["timestamp"]))

        return events[-limit:] if limit else events


class EventSourcedAggregate:
    """
    Base class for event-sourced aggregates.
    """

    def __init__(self):
        self._changes: List[Event] = []
        self._version = 0

    def apply_event(self, event: Event):
        # Override in subclasses
        self._version += 1

    def get_uncommitted_changes(self) -> List[Event]:
        return self._changes

    def mark_changes_as_committed(self):
        self._changes.clear()

    def load_from_history(self, events: List[Event]):
        for event in events:
            self.apply_event(event)
            self._version += 1
