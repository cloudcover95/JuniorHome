"""JuniorHome compose pack. Intent only. Not a Nintendo product."""

from __future__ import annotations

__version__ = "0.1.0"


def ping():
    return {
        "pack": "junior.trit_stack",
        "version": __version__,
        "authority": "python_sim",
        "ports": ("JuniorBitNetFieldCore", "JuniorAstraReason", "JuniorFable"),
    }
