# path: src/juniorhome/service_container.py
#!/usr/bin/env python3
"""
Service Container

Simple but powerful dependency injection / service locator for JuniorHome.
Makes it easy to register, resolve, and manage dependencies across the stack.
"""

import logging
from typing import Any, Callable, Dict, Type, TypeVar

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")

T = TypeVar("T")


class ServiceContainer:
    """
    Lightweight service container (dependency injection).
    """

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        logging.info("ServiceContainer initialized")

    def register(self, interface: Type[T], implementation: T):
        self._services[interface] = implementation
        logging.debug(f"Registered service: {interface.__name__}")

    def register_factory(self, interface: Type[T], factory: Callable[[], T]):
        self._factories[interface] = factory
        logging.debug(f"Registered factory for: {interface.__name__}")

    def resolve(self, interface: Type[T]) -> T:
        if interface in self._services:
            return self._services[interface]

        if interface in self._factories:
            instance = self._factories[interface]()
            self._services[interface] = instance  # Cache it
            return instance

        raise KeyError(f"No registration found for {interface.__name__}")

    def has(self, interface: Type[T]) -> bool:
        return interface in self._services or interface in self._factories

    def clear(self):
        self._services.clear()
        self._factories.clear()
        logging.info("ServiceContainer cleared")
