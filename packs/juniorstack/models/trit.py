"""Ternary pack for JuniorStack status lines. Local only."""
from __future__ import annotations

def pack_status(text: str) -> str:
    """Winsor-style clip to {-1,0,1} on char codes, then I2_S-style hex. Short notes only."""
    if not text:
        return "00"
    vals = [max(-1, min(1, (ord(c) % 3) - 1)) for c in text[:64]]
    bits = "".join({"-1": "10", "0": "00", "1": "01"}[str(v)] for v in vals)
    return hex(int(bits, 2))[2:] if bits else "00"

def note_line(status: str, order_id: int) -> dict:
    body = f"{status}:{order_id}"
    return {"text": body, "pack": pack_status(body), "local": True}
