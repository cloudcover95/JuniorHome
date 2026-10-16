"""GGUF -> bitnetd handshake: ROADMAP #2 GGUF, operator sets JUNIOR_GGUF, bots do not download.

Advances JuniorHome docs/ROADMAP.md item 2 (GGUF). Reads the operator-set
JUNIOR_GGUF path in-process, checks only that the file is already on disk
(stat + 4-byte "GGUF" magic), and maps it to the bitnetd handshake on
127.0.0.1:8765, with i2sd 127.0.0.1:8767 listed alongside. No sockets, no
network, no fetch, no model pull, no download. Stdlib only.
Close rule: sha256(prev || pack || text)[:16] with balanced trits packed as
2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded.
web3node/probe_future.py is reference only and is not imported or wrapped.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
DOWNLOAD = False
MODEL_PULL = False
BIND = "127.0.0.1"
BITNETD_PORT = 8765
I2SD_PORT = 8767
ENV_KEY = "JUNIOR_GGUF"
GGUF_MAGIC = b"GGUF"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "GGUF — operator sets JUNIOR_GGUF. Bots do not download."
DAY = "2026-10-16"

_ROLES = {
    "Home": "operator sets JUNIOR_GGUF",
    "OSai": "reads gguf state",
    "FieldCore": "jsonl intent",
    "Flagstaff": "6-vote AND",
    "JuniorLLM": "bitnetd consumer",
}
_TRITS = {
    "Home": (1, -1, 0, 1),
    "OSai": (0, 1, 1, -1),
    "FieldCore": (-1, 0, 1, 0, 1),
    "Flagstaff": (1, 0, 1, -1, 0, 1),
    "JuniorLLM": (0, -1, 1, 1, -1),
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


def gguf_state(env=None) -> dict:
    """Operator-set path only. Local stat + magic read; never downloads."""
    env = os.environ if env is None else env
    raw = (env.get(ENV_KEY) or "").strip()
    state = {
        "env": ENV_KEY,
        "set": bool(raw),
        "path": raw or None,
        "exists": False,
        "magic_ok": False,
        "download": DOWNLOAD,
        "model_pull": MODEL_PULL,
    }
    if not raw:
        state["reason"] = "unset: operator must set JUNIOR_GGUF"
        return state
    p = Path(raw).expanduser()
    if not p.is_file():
        state["reason"] = "missing on disk: bots do not download"
        return state
    state["exists"] = True
    try:
        with p.open("rb") as fh:
            state["magic_ok"] = fh.read(4) == GGUF_MAGIC
    except OSError:
        state["reason"] = "unreadable"
        return state
    state["reason"] = "ready" if state["magic_ok"] else "not a GGUF file"
    return state


def port_map(state: dict) -> dict:
    """Conceptual map: GGUF path -> bitnetd :8765 on loopback (never opened)."""
    ready = bool(state["exists"] and state["magic_ok"])
    return {
        "bitnetd": {
            "bind": BIND,
            "port": BITNETD_PORT,
            "endpoint": f"http://{BIND}:{BITNETD_PORT}",
            "gguf": state["path"] if ready else None,
            "ready": ready,
            "socket": SOCKET,
            "role": "local bitnet daemon (loads JUNIOR_GGUF)",
        },
        "i2sd": {
            "bind": BIND,
            "port": I2SD_PORT,
            "endpoint": f"http://{BIND}:{I2SD_PORT}",
            "socket": SOCKET,
            "role": "userspace i2s peer",
        },
        **{p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET} for p in PORTS},
    }


def _tag(port: str, ready: bool) -> str:
    return (
        f"gguf-bitnetd:{DAY}:{port}:{_ROLES[port]}:env={ENV_KEY}:ready={ready}:"
        f"bind={BIND}:{BITNETD_PORT}:download={DOWNLOAD}:rate={RATE}"
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


def handshake_check(env=None) -> dict:
    """One side-commit per named port over the GGUF->bitnetd map. In-process only."""
    st = gguf_state(env)
    pm = port_map(st)
    ready = pm["bitnetd"]["ready"]
    chain = build_chain((_TRITS[p], _tag(p, ready)) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": v["ok"] and len(chain) == len(PORTS),
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "gguf": st,
        "bitnetd": pm["bitnetd"],
        "i2sd": pm["i2sd"],
        "ports": list(PORTS),
        "chain": chain,
        "tip": v.get("tip"),
        "close": CLOSE,
    }


def self_check() -> dict:
    import tempfile

    assert BIND == "127.0.0.1"
    assert BITNETD_PORT == 8765 and I2SD_PORT == 8767
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert not (DOWNLOAD or MODEL_PULL)
    assert RATE == 0.05
    assert "JUNIOR_GGUF" in ROADMAP_ITEM and "do not download" in ROADMAP_ITEM
    for p in PORTS:
        t = list(_TRITS[p])
        assert unpack(pack(t), len(t)) == t

    unset = handshake_check({})
    assert unset["ok"] and unset["gguf"]["set"] is False
    assert unset["bitnetd"]["ready"] is False and unset["bitnetd"]["gguf"] is None

    missing = handshake_check({ENV_KEY: "/nonexistent/junior/model.gguf"})
    assert missing["gguf"]["exists"] is False and missing["bitnetd"]["ready"] is False
    assert "do not download" in missing["gguf"]["reason"]

    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "m.gguf"
        good.write_bytes(GGUF_MAGIC + b"\x03\x00\x00\x00")
        bad = Path(d) / "x.bin"
        bad.write_bytes(b"NOPE")
        ok = handshake_check({ENV_KEY: str(good)})
        assert ok["ok"] and ok["bitnetd"]["ready"] is True
        assert ok["bitnetd"]["gguf"] == str(good)
        nb = handshake_check({ENV_KEY: str(bad)})
        assert nb["bitnetd"]["ready"] is False and nb["gguf"]["reason"] == "not a GGUF file"
        assert ok["tip"] != unset["tip"]

    for hs in (unset, missing, ok):
        assert hs["bitnetd"]["endpoint"] == "http://127.0.0.1:8765"
        assert hs["i2sd"]["endpoint"] == "http://127.0.0.1:8767"
        assert all("0.0.0.0" not in r["text"] for r in hs["chain"])
        assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
        assert all("download=False" in r["text"] for r in hs["chain"])
        assert len(hs["tip"]) == 16
        assert verify_chain(hs["chain"])["ok"]

    t = [dict(r) for r in ok["chain"]]
    t[2]["text"] = t[2]["text"].replace("ready=True", "ready=False")
    assert verify_chain(t) == {"ok": False, "at": 2, "reason": "commit"}
    t = [dict(r) for r in ok["chain"]]
    t[4]["trits"] = [0] * len(t[4]["trits"])
    assert verify_chain(t) == {"ok": False, "at": 4, "reason": "commit"}

    live = handshake_check()
    return {
        "ok": True,
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "rate": RATE,
        "close": CLOSE,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "i2sd": f"{BIND}:{I2SD_PORT}",
        "live_gguf_set": live["gguf"]["set"],
        "live_ready": live["bitnetd"]["ready"],
        "tip": live["tip"],
    }


if __name__ == "__main__":
    print(self_check())
