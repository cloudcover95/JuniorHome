"""JTR1 tritpack. GGUF-shaped, not llama.cpp."""
from __future__ import annotations
import json, struct
from pathlib import Path
import numpy as np
from bitnet_engine import infer
from bitnet_orig import absmean
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from second_brain import write_note
from trit_cache import pack_trits
MAGIC, VER = b"JTR1", 1

def write_pack(weights, path=None):
    q, g = absmean(weights)
    blob = pack_trits(np.clip(np.rint(np.asarray(q)), -1, 1))
    path = path or Path(__file__).resolve().parent / "vault" / "weights.jtr1"
    path.write_bytes(MAGIC + struct.pack("<IdII", VER, float(g), len(q), len(blob)) + blob)
    return {"path": str(path), "gamma": g, "n": len(q), "bytes": path.stat().st_size, "gguf_llama": False}

def read_pack(path):
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError("not JTR1")
    ver, g, n, nb = struct.unpack_from("<IdII", raw, 4)
    return {"ver": ver, "gamma": g, "n": n, "payload": raw[24:24+nb]}

def workflow(n=32):
    rng = np.random.default_rng(5)
    w, x = rng.normal(size=n).tolist(), rng.normal(size=n).tolist()
    eng = infer(x, w)
    packed = write_pack(w)
    tf = inject("juniorosai field tritpack saas flagstaff")
    note = write_note(Path(__file__).resolve().parent / "vault", f"# tritpack\n{packed['path']}\n")
    return {"saas": "local-open", "format": "JTR1", "llama_cpp": False,
            "engine": {k: eng[k] for k in ("y", "us", "pack_bytes")}, "file": packed,
            "roundtrip_n": read_pack(packed["path"])["n"],
            "port": pick_juniorllm_port("juniorosai field tritpack"),
            "tf": tf.get("port"), "brain": str(note)}

if __name__ == "__main__":
    print(json.dumps(workflow(), indent=2))
