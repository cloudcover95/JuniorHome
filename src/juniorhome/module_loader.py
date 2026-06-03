# path: src/juniorhome/module_loader.py
#!/usr/bin/env python3
"""
Module Loader

Dynamic module loading system for JuniorHome.
Supports runtime loading of Python modules and packages using importlib.
Serves as the foundation for the mutability layer (JuniorPython-Suite).
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ModuleLoader:
    """
    Dynamic module loader with support for file paths and package names.
    Enables runtime extensibility for the sovereign edge runtime.
    """

    def __init__(self, search_paths: Optional[list] = None):
        self.search_paths = search_paths or []
        self.loaded_modules: Dict[str, Any] = {}
        logging.info("ModuleLoader initialized")

    def load_from_file(self, file_path: str, module_name: Optional[str] = None) -> Any:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Module file not found: {file_path}")

        name = module_name or path.stem

        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        self.loaded_modules[name] = module
        logging.info(f"Loaded module from file: {name}")
        return module

    def load_package(self, package_name: str) -> Any:
        try:
            module = importlib.import_module(package_name)
            self.loaded_modules[package_name] = module
            logging.info(f"Loaded package: {package_name}")
            return module
        except ImportError as e:
            logging.error(f"Failed to load package {package_name}: {e}")
            raise

    def get_module(self, name: str) -> Optional[Any]:
        return self.loaded_modules.get(name)

    def list_loaded(self) -> list:
        return list(self.loaded_modules.keys())

    def reload(self, name: str) -> Any:
        if name not in self.loaded_modules:
            raise KeyError(f"Module {name} not loaded")

        module = self.loaded_modules[name]
        importlib.reload(module)
        logging.info(f"Reloaded module: {name}")
        return module
