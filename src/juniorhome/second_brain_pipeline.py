# path: src/juniorhome/second_brain_pipeline.py
#!/usr/bin/env python3
"""
Second Brain Pipeline (Edge Efficient + CQRS)

Now integrated with EdgeComputeManager for efficient execution
on sovereign edge devices.
"""

import logging

from .cqrs import CommandBus, ProcessRssFeedCommand
from .event_sourcing import EventStore
from .event_bus import EventBus
from .second_brain import SecondBrain
from .command_handlers import ProcessRssFeedHandler, SecondBrainProjection
from .edge_compute_manager import EdgeComputeManager

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SecondBrainPipeline:
    def __init__(self, second_brain: SecondBrain):
        self.second_brain = second_brain
        self.event_store = EventStore()
        self.event_bus = EventBus()
        self.command_bus = CommandBus()
        self.edge_manager = EdgeComputeManager(prefer_quantized=True)
        self.projection = SecondBrainProjection()

        handler = ProcessRssFeedHandler(
            second_brain=self.second_brain,
            event_bus=self.event_bus,
            event_store=self.event_store,
        )
        self.command_bus.register(ProcessRssFeedCommand, handler)

        self.event_bus.subscribe("RssFeedProcessed", self.projection.handle_event)

        logging.info("SecondBrainPipeline initialized (edge-efficient + fully pipelined)")

    def process_rss_feed(self, url: str, tags: list = None):
        # Use edge-efficient execution path
        def _do_process():
            command = ProcessRssFeedCommand(url=url, tags=tags or [])
            return self.command_bus.dispatch(command)

        return self.edge_manager.execute_efficiently(
            _do_process,
            use_quantized=True,
            batch=False,
        )

    def get_stats(self):
        return {
            **self.projection.get_stats(),
            "edge_compute": self.edge_manager.get_stats(),
        }

    def get_recent_events(self, limit: int = 50):
        return self.event_store.get_events(limit=limit)
