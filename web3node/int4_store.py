"""INT4 nibble pack beside trit."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from bitnet_orig import absmean
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from second_brain import write_note
from trit_cache import pack_trits

def to_int4(x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    m = float(np.max(np.abs(a))) or 1.0
    return np.clip(np.rint(a / m * 7.0), -8, 7).astype(np.int8)

def pack_int4(q):
    q = np.clip(q.reshape(-1), -8, 7).astype(np.int8)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
    hi = (q[0::2].astype(np.int16) & 0x0F) << 4
    lo = q[1::2].astype(np.int16) & 0x0F
    return np.array([q.size], dtype=np.uint32).tobytes() + (hi | lo).astype(np.uint8).tobytes()

def train_step(batch):
    flat = [v for row in batch for v in row]
    _q, g = absmean(flat)
    return {"gamma": g, "ste": False, "optimizer": "absmean-refresh"}

def report(n=64):
    x = np.random.default_rng(2).normal(size=n)
    p4 = pack_int4(to_int4(x))
    qt, g = absmean(x.tolist())
    p2 = pack_trits(np.clip(np.rint(np.asarray(qt)), -1, 1))
    tf = inject("juniorosai field int4 flagstaff")
    note = write_note(Path(__file__).resolve().parent / "vault",
                      f"# int4 vs trit\nint4={len(p4)}B trit={len(p2)}B\n")
    return {"n": n, "float32": n * 4, "int4_bytes": len(p4), "trit_bytes": len(p2),
            "gamma": g, "train": train_step([x.tolist()]),
            "port": pick_juniorllm_port("juniorosai field int4 flagstaff"),
            "tf": tf.get("port"), "brain": str(note), "gguf_q4k": False}

if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=2))
