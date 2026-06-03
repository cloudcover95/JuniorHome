# path: src/juniorhome/plugin_loader.py
#!/usr/bin/env python3
"""
JuniorHome Plugin Loader

Dynamic plugin loading system for extensibility.
Allows loading custom modules, agents, or sensors at runtime.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class PluginLoader:
    """
    Production-grade dynamic plugin loader.
    """

    def __init__(self, plugin_paths: Optional[List[str]] = None):
        self.plugin_paths = plugin_paths or []
        self.loaded_plugins: Dict[str, Any] = {}

    def load_plugin(self, module_path: str, class_name: Optional[str] = None) -> Any:
        try:
            module = importlib.import_module(module_path)
            if class_name:
                plugin_class = getattr(module, class_name)
                instance = plugin_class()
                self.loaded_plugins[class_name] = instance
                logging.info(f"Loaded plugin: {class_name} from {module_path}")
                return instance
            else:
                self.loaded_plugins[module_path] = module
                logging.info(f"Loaded module: {module_path}")
                return module
        except Exception as e:
            logging.error(f"Failed to load plugin {module_path}: {e}")
            raise

    def get_plugin(self, name: str) -> Optional[Any]:
        return self.loaded_plugins.get(name)

    def list_plugins(self) -> List[str]:
        return list(self.loaded_plugins.keys())
