# path: src/juniorhome/event_bus.py
#!/usr/bin/env python3
"""
Event Bus

Simple but production-useful event bus for decoupled communication
between components in JuniorHome.
Supports publish/subscribe pattern with topic-based routing.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EventBus:
    """
    Lightweight event bus for pub/sub communication.
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        logging.info("EventBus initialized")

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        self.subscribers[topic].append(callback)
        logging.debug(f"Subscribed to topic: {topic}")

    def unsubscribe(self, topic: str, callback: Callable[[Any], None]):
        if topic in self.subscribers:
            try:
                self.subscribers[topic].remove(callback)
                logging.debug(f"Unsubscribed from topic: {topic}")
            except ValueError:
                pass

    def publish(self, topic: str, data: Any = None):
        if topic not in self.subscribers:
            return

        logging.debug(f"Publishing to topic '{topic}' with {len(self.subscribers[topic])} subscribers")

        for callback in self.subscribers[topic][:]:  # Copy to avoid modification during iteration
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in subscriber for topic '{topic}': {e}")

    def get_subscribers(self, topic: str) -> List[Callable]:
        return self.subscribers.get(topic, [])

    def clear_topic(self, topic: str):
        if topic in self.subscribers:
            self.subscribers[topic].clear()

    def clear_all(self):
        self.subscribers.clear()
        logging.info("EventBus cleared")
