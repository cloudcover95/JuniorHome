"""Daily handshake check across Home, OSai, FieldCore, Flagstaff, JuniorLLM.

Additive ports check for the daily automation. Pure in-process: no sockets,
no network, no fetch. Close rule: sha256(prev || pack || text)[:16] with
balanced trits packed as 2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded.
Stdlib only.
"""
from __future__ import annotations

import hashlib

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
DAY = "2026-10-15"

_ROLES = {
    "Home": "operator box",
    "OSai": "agent layer",
    "FieldCore": "jsonl intent",
    "Flagstaff": "6-vote AND",
    "JuniorLLM": "ticket/ports",
}
# daily trit words (balanced -1/0/1), fixed for reproducible tip
_TRITS = {
    "Home": (1, -1, 0, 1),
    "OSai": (0, 1, 1, -1),
    "FieldCore": (-1, 0, 1, 0, 1),
    "Flagstaff": (1, 1, -1, 1, 1, 0),
    "JuniorLLM": (0, -1, 1, -1, 0),
}

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack(trits) -> bytes:
    """Balanced trits as 2-bit i2s codes, MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack(data: bytes, n: int) -> list[int]:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def side_commit(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def port_map() -> dict:
    """Port name -> role + loopback bind (never opened)."""
    return {p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET} for p in PORTS}


def _tag(port: str) -> str:
    return f"daily-handshake:{DAY}:{port}:{_ROLES[port]}:rate={RATE}"


def build_chain(links, prev: str = GENESIS) -> list[dict]:
    out = []
    for trits, text in links:
        c = side_commit(prev, pack(trits), text)
        out.append({"prev": prev, "trits": list(trits), "text": text, "commit": c})
        prev = c
    return out


def verify_chain(chain, prev: str = GENESIS) -> dict:
    for i, link in enumerate(chain):
        if link["prev"] != prev:
            return {"ok": False, "at": i, "reason": "prev"}
        if side_commit(prev, pack(link["trits"]), link["text"]) != link["commit"]:
            return {"ok": False, "at": i, "reason": "commit"}
        prev = link["commit"]
    return {"ok": True, "n": len(chain), "tip": prev}


def handshake_check() -> dict:
    """One side-commit per port in PORTS order; verify close chain."""
    chain = build_chain((_TRITS[p], _tag(p)) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": v["ok"] and len(chain) == len(PORTS),
        "day": DAY,
        "ports": list(PORTS),
        "chain": chain,
        "tip": v.get("tip"),
        "close": CLOSE,
    }


def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert RATE == 0.05
    pm = port_map()
    assert tuple(pm) == PORTS and all(v["bind"] == BIND for v in pm.values())
    for p in PORTS:
        t = list(_TRITS[p])
        assert unpack(pack(t), len(t)) == t
    pk = pack(_TRITS["Home"])
    c0 = side_commit(GENESIS, pk, _tag("Home"))
    assert len(c0) == 16 and all(ch in "0123456789abcdef" for ch in c0)
    assert c0 == side_commit(GENESIS, pk, _tag("Home"))
    hs = handshake_check()
    assert hs["ok"] and hs["ports"] == list(PORTS)
    assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
    assert len(hs["tip"]) == 16
    bad = [dict(r) for r in hs["chain"]]
    bad[1]["text"] = bad[1]["text"].replace("OSai", "OS0i")
    assert verify_chain(bad) == {"ok": False, "at": 1, "reason": "commit"}
    bad = [dict(r) for r in hs["chain"]]
    bad[4]["trits"] = [0] * len(bad[4]["trits"])
    assert verify_chain(bad) == {"ok": False, "at": 4, "reason": "commit"}
    return {
        "ok": True,
        "rate": RATE,
        "close": CLOSE,
        "bind": BIND,
        "day": DAY,
        "ports": list(PORTS),
        "tip": hs["tip"],
    }


if __name__ == "__main__":
    print(self_check())
