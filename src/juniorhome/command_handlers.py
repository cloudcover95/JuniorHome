# path: src/juniorhome/command_handlers.py
#!/usr/bin/env python3
"""
Command Handlers (Event Sourcing + CQRS)

Example handlers that execute commands, update the Second Brain,
and publish events via EventBus.
"""

import logging
from typing import Any

from .cqrs import CommandHandler, ProcessRssFeedCommand
from .second_brain import SecondBrain
from .event_bus import EventBus
from .event_sourcing import Event, EventStore

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ProcessRssFeedHandler(CommandHandler):
    def __init__(self, second_brain: SecondBrain, event_bus: EventBus, event_store: EventStore):
        self.second_brain = second_brain
        self.event_bus = event_bus
        self.event_store = event_store

    def handle(self, command: ProcessRssFeedCommand) -> Any:
        logging.info(f"Handling ProcessRssFeedCommand for {command.url}")

        # Execute domain logic
        count = 0
        # In real implementation, use RSSIngester here
        # For now we simulate
        finding = {
            "source": "rss",
            "url": command.url,
            "tags": command.tags,
            "processed_at": __import__("time").time(),
        }
        self.second_brain.store_finding(finding)
        count = 1

        # Create and store domain event
        event = Event(
            event_type="RssFeedProcessed",
            data={
                "url": command.url,
                "items_processed": count,
                "tags": command.tags,
            },
        )
        self.event_store.append(event)

        # Publish to EventBus for other components (agents, projections, etc.)
        self.event_bus.publish("RssFeedProcessed", event.data)

        return {"items_processed": count}


class SecondBrainProjection:
    """
    Simple read model / projection for the Second Brain.
    In real CQRS this would be updated by event handlers.
    """

    def __init__(self):
        self.total_findings = 0
        self.sources: dict = {}

    def handle_event(self, event_type: str, data: dict):
        if event_type == "RssFeedProcessed":
            self.total_findings += data.get("items_processed", 0)
            source = data.get("url", "unknown")
            self.sources[source] = self.sources.get(source, 0) + data.get("items_processed", 0)

    def get_stats(self) -> dict:
        return {
            "total_findings": self.total_findings,
            "sources": self.sources,
        }
