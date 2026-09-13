"""BitNet sizing. bytes ≈ params * bits / 8."""
from __future__ import annotations

def weight_bytes(params: int, bits: float) -> int:
    return int(params * bits / 8.0)

def sku(name, params, bits, extra_mb=0.0):
    raw = weight_bytes(params, bits)
    return {"name": name, "params": params, "bits": bits,
            "weight_mib": round(raw / 1024 / 1024, 3),
            "total_mib": round(raw / 1024 / 1024 + extra_mb, 3)}

LADDER = [
    sku("JuniorOSai-T0-kernel", 0, 0),
    sku("ticket-64x64", 64 * 64, 2),
    sku("BitNet-2B4T-I2S", 2_000_000_000, 2),
    sku("Q4-4B", 4_000_000_000, 4.5),
    sku("fp16-2B", 2_000_000_000, 16),
]

def compare():
    return {"live": "kernel + 1 KiB ticket, 0 GB download", "ladder": LADDER,
            "fp16_2B_vs_i2s_2B": {"ratio": 8.0}}

if __name__ == "__main__":
    import json
    print(json.dumps(compare(), indent=2))
