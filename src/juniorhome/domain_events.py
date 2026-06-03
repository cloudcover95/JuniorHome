# path: src/juniorhome/domain_events.py
#!/usr/bin/env python3
"""
Domain Events

Formal domain events for the Second Brain and knowledge domain.
These are different from integration events.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import time


@dataclass
class DomainEvent:
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FindingStored(DomainEvent):
    def __post_init__(self):
        self.event_type = "FindingStored"


@dataclass
class RssFeedProcessed(DomainEvent):
    def __post_init__(self):
        self.event_type = "RssFeedProcessed"


@dataclass
class SecondBrainQueried(DomainEvent):
    def __post_init__(self):
        self.event_type = "SecondBrainQueried"
