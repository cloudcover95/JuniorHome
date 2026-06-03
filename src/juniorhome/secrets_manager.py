# path: src/juniorhome/secrets_manager.py
#!/usr/bin/env python3
"""
Secrets Manager

Secure handling of sensitive configuration (API keys, tokens, passwords).
Supports environment variables and optional encrypted storage.
Production-grade secret management for the sovereign stack.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SecretsManager:
    """
    Manages sensitive secrets securely.
    """

    def __init__(self, secrets_file: Optional[str] = None, encryption_key: Optional[str] = None):
        self.secrets: Dict[str, str] = {}
        self.secrets_file = Path(secrets_file) if secrets_file else None
        self.fernet = None

        if encryption_key and HAS_CRYPTO:
            self.fernet = Fernet(encryption_key.encode())

        self._load_from_env()
        if self.secrets_file:
            self._load_from_file()

        logging.info("SecretsManager initialized")

    def _load_from_env(self):
        for key, value in os.environ.items():
            if key.startswith("SECRET_"):
                secret_name = key[7:].lower()
                self.secrets[secret_name] = value

    def _load_from_file(self):
        if not self.secrets_file or not self.secrets_file.exists():
            return

        try:
            with open(self.secrets_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        self.secrets[key.strip().lower()] = value.strip()
        except Exception as e:
            logging.error(f"Failed to load secrets file: {e}")

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.secrets.get(name.lower(), default)

    def set(self, name: str, value: str):
        self.secrets[name.lower()] = value

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else self.secrets_file
        if not save_path:
            logging.warning("No secrets file path provided")
            return

        try:
            with open(save_path, "w") as f:
                for key, value in self.secrets.items():
                    f.write(f"{key}={value}\n")
            logging.info(f"Saved secrets to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save secrets: {e}")

    def encrypt_value(self, value: str) -> Optional[str]:
        if not self.fernet:
            logging.warning("Encryption not available (cryptography not installed)")
            return None
        return self.fernet.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> Optional[str]:
        if not self.fernet:
            return None
        try:
            return self.fernet.decrypt(encrypted_value.encode()).decode()
        except Exception:
            return None
