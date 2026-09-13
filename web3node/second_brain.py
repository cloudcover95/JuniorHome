"""Obsidian-shaped second brain pulse. Live vault_bridge wins on PYTHONPATH."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from ph_algorithms import h0_persistence, named_algorithms
from tnn_layer import bitlinear

def write_note(vault: Path, body: str) -> Path:
    try:
        import importlib
        return importlib.import_module("obsidian.vault_bridge").write_note(vault, "second_brain.md", body)
    except Exception:
        path = vault / "second_brain.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

def pulse(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    ph = h0_persistence(rng.normal(size=(20, 4)))
    layer = bitlinear(rng.normal(size=32).tolist(), rng.normal(size=32).tolist())
    tf = inject("second brain palace field obsidian")
    port = pick_juniorllm_port("field second brain")
    body = "# Second brain pulse\n\nport: `" + port + "`\n"
    note = write_note(Path(__file__).resolve().parent / "vault", body)
    row = {"port": port, "ph": ph, "tnn": {k: layer[k] for k in ("y", "acc", "dw", "dx", "n")},
           "terraform": {k: tf.get(k) for k in ("port", "ok", "fusion_backend")},
           "note": str(note), "algorithms": named_algorithms()}
    (Path(__file__).resolve().parent / "vault" / "second_brain.json").write_text(
        json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row

if __name__ == "__main__":
    print(json.dumps(pulse(), indent=2, default=str))
