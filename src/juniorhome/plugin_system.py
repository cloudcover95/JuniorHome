# path: src/juniorhome/plugin_system.py
#!/usr/bin/env python3
"""
Plugin System

Advanced plugin management system for JuniorHome.
Builds on ModuleLoader and ServiceRegistry to provide a full
plugin lifecycle (load, register, enable, disable).
"""

import logging
from typing import Any, Dict, List, Optional

from .module_loader import ModuleLoader
from .service_registry import ServiceRegistry

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class PluginSystem:
    """
    Manages the full lifecycle of plugins for the JuniorHome runtime.
    """

    def __init__(self, module_loader: Optional[ModuleLoader] = None, service_registry: Optional[ServiceRegistry] = None):
        self.module_loader = module_loader or ModuleLoader()
        self.service_registry = service_registry or ServiceRegistry()
        self.plugins: Dict[str, Dict[str, Any]] = {}
        logging.info("PluginSystem initialized")

    def load_plugin(self, plugin_path: str, plugin_name: Optional[str] = None) -> bool:
        try:
            module = self.module_loader.load_from_file(plugin_path, plugin_name)
            name = plugin_name or module.__name__

            # Look for a standard entry point
            plugin_instance = None
            if hasattr(module, "create_plugin"):
                plugin_instance = module.create_plugin()
            elif hasattr(module, "Plugin"):
                plugin_instance = module.Plugin()

            self.plugins[name] = {
                "module": module,
                "instance": plugin_instance,
                "path": plugin_path,
                "enabled": True,
            }

            if plugin_instance:
                self.service_registry.register(name, plugin_instance)

            logging.info(f"Plugin loaded: {name}")
            return True

        except Exception as e:
            logging.error(f"Failed to load plugin {plugin_path}: {e}")
            return False

    def enable_plugin(self, name: str) -> bool:
        if name in self.plugins:
            self.plugins[name]["enabled"] = True
            logging.info(f"Plugin enabled: {name}")
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        if name in self.plugins:
            self.plugins[name]["enabled"] = False
            logging.info(f"Plugin disabled: {name}")
            return True
        return False

    def get_plugin(self, name: str) -> Optional[Any]:
        plugin = self.plugins.get(name)
        if plugin and plugin.get("enabled"):
            return plugin.get("instance")
        return None

    def list_plugins(self) -> List[str]:
        return list(self.plugins.keys())

    def unload_plugin(self, name: str) -> bool:
        if name in self.plugins:
            self.service_registry.remove(name)
            del self.plugins[name]
            logging.info(f"Plugin unloaded: {name}")
            return True
        return False
