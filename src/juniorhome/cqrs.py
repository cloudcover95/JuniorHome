# path: src/juniorhome/cqrs.py
#!/usr/bin/env python3
"""
CQRS Foundation

Lightweight Command Query Responsibility Segregation layer.
Works alongside Event Sourcing and the existing EventBus.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Command(ABC):
    pass


class Query(ABC):
    pass


class CommandHandler(ABC):
    @abstractmethod
    def handle(self, command: Command) -> Any:
        pass


class QueryHandler(ABC):
    @abstractmethod
    def handle(self, query: Query) -> Any:
        pass


class CommandBus:
    """
    Simple in-memory Command Bus.
    """

    def __init__(self):
        self._handlers: Dict[Type[Command], CommandHandler] = {}

    def register(self, command_type: Type[Command], handler: CommandHandler):
        self._handlers[command_type] = handler
        logging.debug(f"Registered handler for {command_type.__name__}")

    def dispatch(self, command: Command) -> Any:
        handler = self._handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler registered for {type(command).__name__}")
        return handler.handle(command)


class QueryBus:
    """
    Simple in-memory Query Bus.
    """

    def __init__(self):
        self._handlers: Dict[Type[Query], QueryHandler] = {}

    def register(self, query_type: Type[Query], handler: QueryHandler):
        self._handlers[query_type] = handler
        logging.debug(f"Registered handler for {query_type.__name__}")

    def dispatch(self, query: Query) -> Any:
        handler = self._handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler registered for {type(query).__name__}")
        return handler.handle(query)


# Example Commands and Queries for Second Brain

class ProcessRssFeedCommand(Command):
    def __init__(self, url: str, tags: list = None):
        self.url = url
        self.tags = tags or []


class QuerySecondBrainQuery(Query):
    def __init__(self, topic: str):
        self.topic = topic
