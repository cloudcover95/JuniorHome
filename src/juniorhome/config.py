# path: src/juniorhome/config.py
#!/usr/bin/env python3
"""
JuniorHome Configuration System

Production-grade config loading with Pydantic models.
Supports YAML/JSON with Python 3.9 compatibility.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = lambda **kwargs: None


class JuniorHomeConfig:
    """
    Simple but production-ready configuration holder.
    Falls back gracefully if Pydantic is not available.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or {}

        # Core settings
        self.workspace_root = self.config.get("workspace_root", str(Path.home() / "JuniorCloud" / "juniorstock"))
        self.log_level = self.config.get("log_level", "INFO")
        self.enable_bitnet = self.config.get("enable_bitnet", True)
        self.enable_swarm = self.config.get("enable_swarm", True)
        self.batch_size = self.config.get("batch_size", 50)
        self.flush_interval = self.config.get("flush_interval", 30.0)

    @classmethod
    def from_file(cls, path: str) -> "JuniorHomeConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls({})

        if config_path.suffix in [".yaml", ".yml"] and HAS_YAML:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            with open(config_path) as f:
                data = json.load(f)

        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "log_level": self.log_level,
            "enable_bitnet": self.enable_bitnet,
            "enable_swarm": self.enable_swarm,
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
        }
