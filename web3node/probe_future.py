"""Read-only probes for GGUF / i2sd / Asahi / UE5. No downloads."""
from __future__ import annotations

import json
import os
from pathlib import Path


def probe() -> dict:
    gguf = os.environ.get("JUNIOR_GGUF", "")
    return {
        "gguf_path": gguf or None,
        "gguf_exists": bool(gguf) and Path(gguf).is_file(),
        "llama_ready": bool(gguf) and Path(gguf).is_file(),
        "i2sd_bind": "127.0.0.1:8767",
        "asahi": Path("/sys/class/drm").exists() and "asahi" in os.uname().release.lower(),
        "ue5_env": os.environ.get("JUNIOR_UE5") == "1",
        "ue5_launch_allowed": False,
        "t_next": "T13",
        "slots_paused": ["07:00", "13:00"],
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
