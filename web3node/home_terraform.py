"""JuniorHome terraform sidecar. Live ports.terraform wins when JuniorLLM is on PYTHONPATH."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from bitnet_orig import absmean, i2s_pack
from fieldcore_bridge import pick_juniorllm_port

DROP = re.compile(
    r"(ignore previous|system prompt|0\.0\.0\.0|wget |curl http)",
    re.I,
)


def terraform(text: str) -> dict:
    try:
        import importlib

        mod = importlib.import_module("ports.terraform")
        return mod.terraform(text)
    except Exception:
        cleaned = DROP.sub("", text or "")
        cleaned = " ".join(cleaned.split())
        xs = [float(ord(c) % 97) for c in (cleaned or "x")[:16]]
        trits, scale = absmean(xs)
        return {
            "port": pick_juniorllm_port(cleaned),
            "text": cleaned,
            "ok": "0.0.0.0" not in cleaned.lower(),
            "fusion_y": scale,
            "fusion_backend": "orig-python",
            "llama_ready": False,
            "i2s": i2s_pack(trits).hex(),
        }


def inject(text: str, vault: Path | None = None) -> dict:
    row = terraform(text)
    row["t"] = int(time.time() * 1000)
    vault = vault or Path(__file__).resolve().parent / "vault" / "terraform.jsonl"
    vault.parent.mkdir(parents=True, exist_ok=True)
    with vault.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    row["vault"] = str(vault)
    return row
