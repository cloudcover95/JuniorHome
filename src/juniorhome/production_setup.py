# path: src/juniorhome/production_setup.py
#!/usr/bin/env python3
"""
Production Setup

Production-grade initialization for JuniorHome.
Sets up structured logging, error handling boundaries, graceful shutdown,
and environment validation. Essential for release-grade deployments.
"""

import atexit
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ProductionSetup:
    """
    Handles production initialization and lifecycle management.
    """

    def __init__(self, app_name: str = "JuniorHome", log_level: str = "INFO", log_file: Optional[str] = None):
        self.app_name = app_name
        self.log_level = log_level.upper()
        self.log_file = log_file
        self._shutdown_handlers: list = []

        self._setup_logging()
        self._setup_signal_handlers()
        self._register_shutdown()

        logging.info(f"{self.app_name} production setup complete")

    def _setup_logging(self):
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        handlers = [logging.StreamHandler(sys.stdout)]

        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))

        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format=log_format,
            handlers=handlers,
            force=True
        )

        # Reduce noise from third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    def _setup_signal_handlers(self):
        def handle_signal(signum, frame):
            logging.warning(f"Received signal {signum}. Initiating graceful shutdown...")
            self.shutdown()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _register_shutdown(self):
        atexit.register(self.shutdown)

    def register_shutdown_handler(self, handler: Callable[[], None]):
        self._shutdown_handlers.append(handler)

    def shutdown(self):
        logging.info(f"Shutting down {self.app_name}...")
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logging.error(f"Shutdown handler failed: {e}")
        logging.info(f"{self.app_name} shutdown complete")

    def validate_environment(self) -> Dict[str, bool]:
        checks = {
            "python_version": sys.version_info >= (3, 9),
            "write_access": os.access(".", os.W_OK),
        }
        return checks

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "python_version": sys.version,
            "platform": sys.platform,
            "log_level": self.log_level,
        }
