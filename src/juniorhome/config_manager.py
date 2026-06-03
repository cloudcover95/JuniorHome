# path: src/juniorhome/config_manager.py
#!/usr/bin/env python3
"""
Config Manager

Production-grade configuration management for JuniorHome.
Supports environment variables, YAML/JSON config files, and validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ConfigManager:
    """
    Centralized configuration with multiple sources and validation.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.config: Dict[str, Any] = {}
        self.config_file = Path(config_file) if config_file else None

        self._load_defaults()
        self._load_from_file()
        self._load_from_env()

        logging.info("ConfigManager initialized")

    def _load_defaults(self):
        self.config = {
            "app_name": "JuniorHome",
            "log_level": "INFO",
            "data_dir": "data",
            "use_ternary": False,
            "ollama_url": "http://localhost:11434",
            "enable_docker": True,
        }

    def _load_from_file(self):
        if not self.config_file or not self.config_file.exists():
            return

        try:
            if self.config_file.suffix in [".yaml", ".yml"] and HAS_YAML:
                with open(self.config_file) as f:
                    file_config = yaml.safe_load(f) or {}
            else:
                with open(self.config_file) as f:
                    file_config = json.load(f)

            self.config.update(file_config)
            logging.info(f"Loaded config from {self.config_file}")
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")

    def _load_from_env(self):
        env_prefix = "JUNIORHOME_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                # Try to convert types
                if value.lower() in ["true", "false"]:
                    self.config[config_key] = value.lower() == "true"
                else:
                    try:
                        self.config[config_key] = int(value)
                    except ValueError:
                        try:
                            self.config[config_key] = float(value)
                        except ValueError:
                            self.config[config_key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else self.config_file
        if not save_path:
            save_path = Path("config.json")

        try:
            with open(save_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"Saved config to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
