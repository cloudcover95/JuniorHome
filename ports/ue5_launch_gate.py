"""UE5 launch gate handshake: ROADMAP #5 UE5 launched, JUNIOR_UE5=1 and watts>=80.

Advances JuniorHome docs/ROADMAP.md item 5 (UE5 launched). Reads the
operator-set JUNIOR_UE5 flag and JUNIOR_WATTS budget in-process, computes
whether the gate condition (JUNIOR_UE5 == "1" and watts >= 80) is met, and
maps it across Home, OSai, FieldCore, Flagstaff and JuniorLLM. The default
route stays FrameForge2D. This check never launches anything: os_route.launch
and ue5_launch are asserted False on every path, including when the gate
condition is met (gate_open only reports eligibility for the operator).
bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767 are listed, never opened.
No sockets, no network, no hardware calls, no fetch, no download. Stdlib only.
Close rule: sha256(prev || pack || text)[:16] with balanced trits packed as
2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded.
web3node/os_route.py is reference only and is not imported or wrapped.
"""
from __future__ import annotations

import hashlib
import math
import os

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
OS_ROUTE_LAUNCH = False
DOWNLOAD = False
HARDWARE_CALL = False
BIND = "127.0.0.1"
BITNETD_PORT = 8765
I2SD_PORT = 8767
ENV_FLAG = "JUNIOR_UE5"
ENV_WATTS = "JUNIOR_WATTS"
WATTS_MIN = 80.0
DEFAULT_ROUTE = "FrameForge2D"
GATED_ROUTE = "UE5"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = (
    "UE5 launched — JUNIOR_UE5=1 and watts>=80. Default FrameForge2D. "
    "os_route.launch false until then."
)
DAY = "2026-10-16"

_ROLES = {
    "Home": "operator sets JUNIOR_UE5 and JUNIOR_WATTS",
    "OSai": "reads gate state, route FrameForge2D default",
    "FieldCore": "jsonl intent",
    "Flagstaff": "6-vote AND",
    "JuniorLLM": "bitnetd consumer, no render",
}
_TRITS = {
    "Home": (1, 0, -1, 1, 1),
    "OSai": (0, 1, -1, 0),
    "FieldCore": (-1, 1, 0, 1, 0),
    "Flagstaff": (1, 1, 0, -1, 0, 1),
    "JuniorLLM": (0, 0, 1, -1, 1),
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


def _watts(raw: str):
    try:
        w = float(raw)
    except (TypeError, ValueError):
        return None
    return w if math.isfinite(w) and w >= 0 else None


def gate_state(env=None) -> dict:
    """Operator-set env only. No hardware power read; never launches."""
    env = os.environ if env is None else env
    flag_raw = (env.get(ENV_FLAG) or "").strip()
    watts_raw = (env.get(ENV_WATTS) or "").strip()
    watts = _watts(watts_raw) if watts_raw else None
    flag_ok = flag_raw == "1"
    watts_ok = watts is not None and watts >= WATTS_MIN
    gate_open = flag_ok and watts_ok
    if not flag_ok:
        reason = "JUNIOR_UE5 != 1: stay FrameForge2D"
    elif watts is None:
        reason = "JUNIOR_WATTS unset or invalid: stay FrameForge2D"
    elif not watts_ok:
        reason = "watts<80: stay FrameForge2D"
    else:
        reason = "eligible: operator launches, bots do not"
    return {
        "flag_env": ENV_FLAG,
        "watts_env": ENV_WATTS,
        "flag": flag_raw or None,
        "watts": watts,
        "watts_min": WATTS_MIN,
        "flag_ok": flag_ok,
        "watts_ok": watts_ok,
        "gate_open": gate_open,
        "route": GATED_ROUTE if gate_open else DEFAULT_ROUTE,
        "os_route_launch": OS_ROUTE_LAUNCH,
        "ue5_launch": UE5_LAUNCH,
        "hardware_call": HARDWARE_CALL,
        "reason": reason,
    }


def port_map(state: dict) -> dict:
    """Conceptual map only: loopback endpoints listed, never opened."""
    return {
        "os_route": {
            "route": state["route"],
            "default": DEFAULT_ROUTE,
            "launch": OS_ROUTE_LAUNCH,
            "gate_open": state["gate_open"],
        },
        "bitnetd": {
            "bind": BIND,
            "port": BITNETD_PORT,
            "endpoint": f"http://{BIND}:{BITNETD_PORT}",
            "socket": SOCKET,
            "role": "local bitnet daemon (independent of render route)",
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


def _tag(port: str, st: dict) -> str:
    return (
        f"ue5-gate:{DAY}:{port}:{_ROLES[port]}:flag={st['flag_ok']}:"
        f"watts={st['watts']}:min={WATTS_MIN}:open={st['gate_open']}:"
        f"route={st['route']}:launch={OS_ROUTE_LAUNCH}:ue5_launch={UE5_LAUNCH}:"
        f"bind={BIND}:{BITNETD_PORT}:{I2SD_PORT}:rate={RATE}"
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
    """One side-commit per named port over the UE5 gate map. In-process only."""
    st = gate_state(env)
    pm = port_map(st)
    chain = build_chain((_TRITS[p], _tag(p, st)) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": v["ok"] and len(chain) == len(PORTS) and pm["os_route"]["launch"] is False,
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "gate": st,
        "os_route": pm["os_route"],
        "bitnetd": pm["bitnetd"],
        "i2sd": pm["i2sd"],
        "ports": list(PORTS),
        "chain": chain,
        "tip": v.get("tip"),
        "close": CLOSE,
    }


def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert BITNETD_PORT == 8765 and I2SD_PORT == 8767
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert not (OS_ROUTE_LAUNCH or DOWNLOAD or HARDWARE_CALL)
    assert RATE == 0.05 and WATTS_MIN == 80.0
    assert "JUNIOR_UE5=1" in ROADMAP_ITEM and "watts>=80" in ROADMAP_ITEM
    for p in PORTS:
        t = list(_TRITS[p])
        assert unpack(pack(t), len(t)) == t

    cases = {
        "unset": ({}, False),
        "flag_only": ({ENV_FLAG: "1"}, False),
        "low_watts": ({ENV_FLAG: "1", ENV_WATTS: "79.9"}, False),
        "bad_watts": ({ENV_FLAG: "1", ENV_WATTS: "nan"}, False),
        "neg_watts": ({ENV_FLAG: "1", ENV_WATTS: "-100"}, False),
        "flag_zero": ({ENV_FLAG: "0", ENV_WATTS: "120"}, False),
        "flag_true_word": ({ENV_FLAG: "true", ENV_WATTS: "120"}, False),
        "edge": ({ENV_FLAG: "1", ENV_WATTS: "80"}, True),
        "high": ({ENV_FLAG: " 1 ", ENV_WATTS: "150"}, True),
    }
    res = {}
    for name, (env, want) in cases.items():
        hs = handshake_check(env)
        res[name] = hs
        g = hs["gate"]
        assert hs["ok"], name
        assert g["gate_open"] is want, name
        assert g["route"] == (GATED_ROUTE if want else DEFAULT_ROUTE), name
        assert hs["os_route"]["launch"] is False, name
        assert g["os_route_launch"] is False and g["ue5_launch"] is False, name
        assert hs["bitnetd"]["endpoint"] == "http://127.0.0.1:8765"
        assert hs["i2sd"]["endpoint"] == "http://127.0.0.1:8767"
        assert all("0.0.0.0" not in r["text"] for r in hs["chain"])
        assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
        assert all("launch=False" in r["text"] for r in hs["chain"])
        assert len(hs["tip"]) == 16 and verify_chain(hs["chain"])["ok"]
    assert res["edge"]["tip"] != res["unset"]["tip"]
    assert res["unset"]["gate"]["reason"].startswith("JUNIOR_UE5 != 1")

    t = [dict(r) for r in res["edge"]["chain"]]
    t[1]["text"] = t[1]["text"].replace("launch=False", "launch=True")
    assert verify_chain(t) == {"ok": False, "at": 1, "reason": "commit"}
    t = [dict(r) for r in res["edge"]["chain"]]
    t[3]["trits"] = [0] * len(t[3]["trits"])
    assert verify_chain(t) == {"ok": False, "at": 3, "reason": "commit"}

    live = handshake_check()
    assert live["os_route"]["launch"] is False
    return {
        "ok": True,
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "rate": RATE,
        "close": CLOSE,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "i2sd": f"{BIND}:{I2SD_PORT}",
        "live_gate_open": live["gate"]["gate_open"],
        "live_route": live["gate"]["route"],
        "os_route_launch": live["os_route"]["launch"],
        "tip": live["tip"],
    }


if __name__ == "__main__":
    print(self_check())
