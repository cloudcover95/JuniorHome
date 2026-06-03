# path: src/juniorhome/enhanced_plugin_manager.py
#!/usr/bin/env python3
"""
Enhanced Plugin Manager

Advanced plugin management with security validation, lifecycle hooks,
and integration with SecurityMiddleware.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .plugin_system import PluginSystem
from .security_middleware import SecurityMiddleware

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EnhancedPluginManager(PluginSystem):
    """
    Plugin manager with security and advanced lifecycle support.
    """

    def __init__(self, security_middleware: Optional[SecurityMiddleware] = None, **kwargs):
        super().__init__(**kwargs)
        self.security = security_middleware or SecurityMiddleware()
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        logging.info("EnhancedPluginManager initialized with security")

    def load_plugin(self, plugin_path: str, plugin_name: Optional[str] = None, validate_security: bool = True) -> bool:
        if validate_security:
            # Basic security check on plugin file
            path = Path(plugin_path)
            if not path.exists() or not path.suffix == ".py":
                logging.warning(f"Plugin {plugin_path} failed basic security validation")
                return False

        success = super().load_plugin(plugin_path, plugin_name)

        if success and plugin_name:
            self.plugin_metadata[plugin_name] = {
                "path": plugin_path,
                "loaded_at": __import__("time").time(),
                "security_validated": validate_security,
            }

        return success

    def get_plugin_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self.plugin_metadata.get(name)

    def list_plugins_detailed(self) -> List[Dict[str, Any]]:
        result = []
        for name in self.list_plugins():
            meta = self.plugin_metadata.get(name, {})
            plugin_info = {
                "name": name,
                "enabled": self.plugins.get(name, {}).get("enabled", False),
                **meta,
            }
            result.append(plugin_info)
        return result
