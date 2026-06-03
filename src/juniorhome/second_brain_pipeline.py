# path: src/juniorhome/second_brain_pipeline.py
#!/usr/bin/env python3
"""
Second Brain Pipeline

Fully integrated pipeline combining:
- CommandBus (CQRS)
- EventStore (Event Sourcing)
- EventBus (pub/sub)
- SecondBrain domain logic
- Projection (read model)

This creates a complete, production-style pipeline for knowledge ingestion and querying.
"""

import logging

from .cqrs import CommandBus, ProcessRssFeedCommand
from .event_sourcing import EventStore
from .event_bus import EventBus
from .second_brain import SecondBrain
from .command_handlers import ProcessRssFeedHandler, SecondBrainProjection

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SecondBrainPipeline:
    """
    End-to-end pipelined system for the Second Brain using
    Event Sourcing + CQRS.
    """

    def __init__(self, second_brain: SecondBrain):
        self.second_brain = second_brain
        self.event_store = EventStore()
        self.event_bus = EventBus()
        self.command_bus = CommandBus()

        # Projection (read model)
        self.projection = SecondBrainProjection()

        # Register handler
        handler = ProcessRssFeedHandler(
            second_brain=self.second_brain,
            event_bus=self.event_bus,
            event_store=self.event_store,
        )
        self.command_bus.register(ProcessRssFeedCommand, handler)

        # Subscribe projection to events
        self.event_bus.subscribe("RssFeedProcessed", self.projection.handle_event)

        logging.info("SecondBrainPipeline initialized (fully pipelined)")

    def process_rss_feed(self, url: str, tags: list = None):
        command = ProcessRssFeedCommand(url=url, tags=tags or [])
        result = self.command_bus.dispatch(command)
        return result

    def get_stats(self):
        return self.projection.get_stats()

    def get_recent_events(self, limit: int = 50):
        return self.event_store.get_events(limit=limit)
