"""i2sd userspace handshake — ROADMAP #3 i2sd userspace 127.0.0.1:8767, no docker.sock.

Advances JuniorHome docs/ROADMAP.md item 3 (i2sd userspace). Pure in-process
handshake + port map: no sockets, no docker.sock, no network, no fetch.
Close rule: sha256(prev || pack || text)[:16] with balanced trits packed as
2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.
"""
from __future__ import annotations

import hashlib

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
DOCKER_SOCK = False
BIND = "127.0.0.1"
I2SD_PORT = 8767
BITNETD_PORT = 8765
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "i2sd userspace — 127.0.0.1:8767. No docker.sock."
DAY = "2026-10-16"

_ROLES = {
    "Home": "operator box",
    "OSai": "agent layer",
    "FieldCore": "jsonl intent",
    "Flagstaff": "6-vote AND",
    "JuniorLLM": "ticket/ports",
}
# fixed trit words for reproducible tip (balanced -1/0/1)
_TRITS = {
    "Home": (1, 0, -1, 1),
    "OSai": (0, 1, -1, 0),
    "FieldCore": (1, -1, 1, 0, -1),
    "Flagstaff": (1, 1, 0, -1, 1, 0),
    "JuniorLLM": (-1, 0, 1, 1, 0),
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
    """Conceptual port map: i2sd :8767 + bitnetd :8765 on loopback only (never opened)."""
    return {
        "i2sd": {
            "bind": BIND,
            "port": I2SD_PORT,
            "endpoint": f"http://{BIND}:{I2SD_PORT}",
            "docker_sock": DOCKER_SOCK,
            "socket": SOCKET,
            "role": "userspace handshake",
        },
        "bitnetd": {
            "bind": BIND,
            "port": BITNETD_PORT,
            "endpoint": f"http://{BIND}:{BITNETD_PORT}",
            "socket": SOCKET,
            "role": "local bitnet daemon",
        },
        **{
            p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET}
            for p in PORTS
        },
    }


def _tag(port: str) -> str:
    return (
        f"i2sd-userspace:{DAY}:{port}:{_ROLES[port]}:"
        f"bind={BIND}:{I2SD_PORT}:docker_sock={DOCKER_SOCK}:rate={RATE}"
    )


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
    """One side-commit per named port; verify close chain. In-process only."""
    chain = build_chain((_TRITS[p], _tag(p)) for p in PORTS)
    v = verify_chain(chain)
    pm = port_map()
    return {
        "ok": v["ok"] and len(chain) == len(PORTS),
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "ports": list(PORTS),
        "i2sd": pm["i2sd"],
        "bitnetd": pm["bitnetd"],
        "chain": chain,
        "tip": v.get("tip"),
        "close": CLOSE,
    }


def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert I2SD_PORT == 8767
    assert BITNETD_PORT == 8765
    assert DOCKER_SOCK is False
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert RATE == 0.05
    assert "i2sd userspace" in ROADMAP_ITEM
    assert "8767" in ROADMAP_ITEM
    assert "docker.sock" in ROADMAP_ITEM
    pm = port_map()
    assert pm["i2sd"]["bind"] == BIND
    assert pm["i2sd"]["port"] == I2SD_PORT
    assert pm["i2sd"]["docker_sock"] is False
    assert pm["i2sd"]["socket"] is False
    assert pm["bitnetd"]["port"] == BITNETD_PORT
    assert pm["bitnetd"]["bind"] == BIND
    assert "0.0.0.0" not in pm["i2sd"]["endpoint"]
    assert "0.0.0.0" not in pm["bitnetd"]["endpoint"]
    for p in PORTS:
        t = list(_TRITS[p])
        assert unpack(pack(t), len(t)) == t
        assert pm[p]["bind"] == BIND and pm[p]["socket"] is False
    pk = pack(_TRITS["Home"])
    c0 = side_commit(GENESIS, pk, _tag("Home"))
    assert len(c0) == 16 and all(ch in "0123456789abcdef" for ch in c0)
    assert c0 == side_commit(GENESIS, pk, _tag("Home"))
    hs = handshake_check()
    assert hs["ok"] and hs["ports"] == list(PORTS)
    assert hs["roadmap_item"] == ROADMAP_ITEM
    assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
    assert all(f"bind={BIND}:{I2SD_PORT}" in r["text"] for r in hs["chain"])
    assert all("docker_sock=False" in r["text"] for r in hs["chain"])
    assert len(hs["tip"]) == 16
    bad = [dict(r) for r in hs["chain"]]
    bad[1]["text"] = bad[1]["text"].replace("OSai", "OS0i")
    assert verify_chain(bad) == {"ok": False, "at": 1, "reason": "commit"}
    bad = [dict(r) for r in hs["chain"]]
    bad[3]["trits"] = [0] * len(bad[3]["trits"])
    assert verify_chain(bad) == {"ok": False, "at": 3, "reason": "commit"}
    return {
        "ok": True,
        "rate": RATE,
        "close": CLOSE,
        "bind": BIND,
        "i2sd_port": I2SD_PORT,
        "docker_sock": DOCKER_SOCK,
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "ports": list(PORTS),
        "tip": hs["tip"],
    }


if __name__ == "__main__":
    print(self_check())
