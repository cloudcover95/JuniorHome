"""UE5 FrameForge2D default handshake — ROADMAP #5 UE5 launched: JUNIOR_UE5=1 and watts>=80. Default FrameForge2D. os_route.launch false until then.

Advances JuniorHome docs/ROADMAP.md item 5 (UE5 launched). Pure in-process
handshake + port map between Home, OSai, FieldCore, Flagstaff and JuniorLLM.
The default route is declarative: it reads a facts dict handed in by the caller
(default: static, FrameForge2D, launch false) and never boots UE, never sets
JUNIOR_UE5, never reads watts live, never opens a socket, never fetches, never
runs a subprocess, never pulls a model. A default-route check only counts when
origin == "operator" and the operator opted in.

Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false, fetch false, layer_one false, ue5_launch false,
bind 127.0.0.1 only. Rate is a constant; no poll sets it.
Close rule: sha256(prev || pack || text)[:16], balanced trits packed as
2-bit codes (0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.
"""
from __future__ import annotations

import hashlib
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
FETCH = False
LAYER_ONE = False
UE5_LAUNCH = False
MODEL_PULL = False
OS_ROUTE_LAUNCH = False
DEFAULT_ROUTE = "FrameForge2D"
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "UE5 launched — JUNIOR_UE5=1 and watts>=80. Default FrameForge2D. os_route.launch false until then."
DAY = "2026-10-18"
PROBE_ORIGIN = "operator"

_ROLES = {
    "Home": "operator box / default FrameForge2D until gate",
    "OSai": "agent layer / route reader",
    "FieldCore": "jsonl intent / route record",
    "Flagstaff": "6-vote AND / route gate",
    "JuniorLLM": "goldens junior_osai/goldens/ue5_default.json",
}
# ONE trit pack for the whole handshake (balanced -1/0/1), reused every link.
TRIT_PACK = (1, 0, -1, 1, 0, -1, 1, 1)

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}

# Static default facts: FrameForge2D, launch false, nothing run. Callers may inject
# their own dict; we never collect live or launch.
DEFAULT_FACTS = {
    "origin": PROBE_ORIGIN,
    "route": DEFAULT_ROUTE,
    "os_route_launch": False,
    "ue5_launch": False,
    "junior_ue5": "0",
    "watts": 0,
}


def pack(trits) -> bytes:
    """Balanced trits as 2-bit codes, MSB first, zero-padded."""
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


def unpack(data: bytes, n: int) -> list:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


PACK = pack(TRIT_PACK)


def close(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def anticon(facts: dict) -> dict:
    """AntiCon gate: reject anything that would widen the path."""
    bad = []
    if facts.get("origin") != PROBE_ORIGIN:
        bad.append("origin")
    for k in ("bind",):
        if k in facts and facts[k] != BIND:
            bad.append("bind")
    for k in ("socket", "fetch", "mint", "layer_one", "ue5_launch",
              "model_pull", "os_route_launch", "rate_poll"):
        if facts.get(k):
            bad.append(k)
    if "rate" in facts and facts["rate"] != RATE:
        bad.append("rate")
    if facts.get("route") and facts["route"] != DEFAULT_ROUTE:
        bad.append("route")
    return {"ok": not bad, "reject": bad}


def probe(facts: dict | None = None) -> dict:
    """Declarative UE5 default FrameForge2D probe. Reads injected facts only; launches nothing."""
    f = dict(DEFAULT_FACTS if facts is None else facts)
    ac = anticon(f)
    ready = bool(f.get("route") == DEFAULT_ROUTE and not f.get("os_route_launch") and not f.get("ue5_launch"))
    return {
        "opt_in": OPT_IN,
        "anticon": ac,
        "counted": OPT_IN and ac["ok"],
        "origin": f.get("origin"),
        "route": f.get("route", DEFAULT_ROUTE),
        "os_route_launch": bool(f.get("os_route_launch")),
        "ue5_launch": bool(f.get("ue5_launch")),
        "default_ready": ready and ac["ok"],
        "model_pull": MODEL_PULL,
        "socket": SOCKET,
    }


def port_map() -> dict:
    """Conceptual loopback port map (never opened)."""
    return {
        p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET, "fetch": FETCH, "ue5_launch": UE5_LAUNCH}
        for p in PORTS
    }


def _text(port: str, pr: dict) -> str:
    return (
        f"ue5-frameforge2d-default:{DAY}:{port}:{_ROLES[port]}:origin={pr['origin']}:"
        f"counted={pr['counted']}:opt_in={OPT_IN}:anticon={pr['anticon']['ok']}:"
        f"pool_held={POOL_HELD}:mint={MINT}:socket={SOCKET}:fetch={FETCH}:"
        f"layer_one={LAYER_ONE}:ue5_launch={UE5_LAUNCH}:os_route_launch={OS_ROUTE_LAUNCH}:"
        f"route={DEFAULT_ROUTE}:bind={BIND}:rate={RATE}"
    )


def build_chain(texts, prev: str = GENESIS) -> list:
    out = []
    for text in texts:
        c = close(prev, PACK, text)
        out.append({"prev": prev, "text": text, "commit": c})
        prev = c
    return out


def verify_chain(chain, prev: str = GENESIS, pk: bytes = PACK) -> dict:
    for i, link in enumerate(chain):
        if link["prev"] != prev:
            return {"ok": False, "at": i, "reason": "prev"}
        if close(prev, pk, link["text"]) != link["commit"]:
            return {"ok": False, "at": i, "reason": "commit"}
        prev = link["commit"]
    return {"ok": True, "n": len(chain), "tip": prev}


def handshake_check(facts: dict | None = None) -> dict:
    pr = probe(facts)
    chain = build_chain(_text(p, pr) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": v["ok"] and len(chain) == len(PORTS) and pr["counted"],
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "ports": list(PORTS),
        "probe": pr,
        "chain": chain,
        "tip": v.get("tip"),
        "close": CLOSE,
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or FETCH or LAYER_ONE or UE5_LAUNCH or OS_ROUTE_LAUNCH)
    assert not MODEL_PULL
    assert BIND == "127.0.0.1"
    assert "UE5 launched" in ROADMAP_ITEM and "FrameForge2D" in ROADMAP_ITEM and "os_route.launch false" in ROADMAP_ITEM
    assert unpack(PACK, len(TRIT_PACK)) == list(TRIT_PACK)
    for p, v in port_map().items():
        assert v["bind"] == BIND and v["socket"] is False and v["fetch"] is False and v["ue5_launch"] is False
    c0 = close(GENESIS, PACK, "x")
    assert c0 == hashlib.sha256(GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    hs = handshake_check()
    assert hs["ok"] and hs["ports"] == list(PORTS)
    assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
    assert all(f"bind={BIND}" in r["text"] for r in hs["chain"])
    assert all("pool_held=True" in r["text"] for r in hs["chain"])
    assert all("ue5_launch=False" in r["text"] for r in hs["chain"])
    assert all("route=FrameForge2D" in r["text"] for r in hs["chain"])
    assert len(hs["tip"]) == 16
    # non-operator probe is not counted
    assert probe({"origin": "manual"})["counted"] is False
    # AntiCon rejects widening attempts
    for k, v in (("socket", True), ("fetch", True), ("mint", True),
                 ("model_pull", True), ("ue5_launch", True), ("rate_poll", True),
                 ("os_route_launch", True), ("rate", 0.5), ("bind", "0.0.0.0"),
                 ("route", "UE5")):
        f = dict(DEFAULT_FACTS, **{k: v})
        assert anticon(f)["ok"] is False, k
        assert handshake_check(f)["ok"] is False, k
    # tamper detection
    bad = [dict(r) for r in hs["chain"]]
    bad[2]["text"] = bad[2]["text"].replace("FieldCore", "FieldC0re")
    assert verify_chain(bad) == {"ok": False, "at": 2, "reason": "commit"}
    assert verify_chain(hs["chain"], pk=pack((0,) * len(TRIT_PACK)))["ok"] is False
    return {
        "ok": True,
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "rate": RATE,
        "close": CLOSE,
        "bind": BIND,
        "opt_in": OPT_IN,
        "anticon": ANTICON,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "fetch": FETCH,
        "layer_one": LAYER_ONE,
        "ue5_launch": UE5_LAUNCH,
        "os_route_launch": OS_ROUTE_LAUNCH,
        "ports": list(PORTS),
        "tip": hs["tip"],
    }


if __name__ == "__main__":
    try:
        print(self_check())
    except AssertionError as e:
        print({"ok": False, "error": repr(e)})
        sys.exit(1)
    sys.exit(0)
