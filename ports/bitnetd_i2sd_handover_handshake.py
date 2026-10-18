"""bitnetd -> i2sd handover handshake — BETA_TO_OS "JuniorOS overlay" item.

Advances JuniorLLM docs/BETA_TO_OS.md item "JuniorOS overlay" (hardened
bitnetd.service + juniorctl on PATH; bitnetd on 127.0.0.1:8765 only), with
JuniorHome docs/ROADMAP.md #3 "i2sd userspace — 127.0.0.1:8767" as the
fallback lane. New angle: the fallback and handover ORDER as seen from
JuniorHome. bitnetd 127.0.0.1:8765 is always primary; i2sd 127.0.0.1:8767 is
only ever the second choice; the order is never reversed and both lanes are
never active at once. If neither lane is ready the handover holds.

Handshake / port map between Home, OSai, FieldCore, Flagstaff and JuniorLLM.
Pure in-process and declarative: it reads a facts dict handed in by the
caller (default: static, everything held) and never opens a socket, binds,
fetches, runs a subprocess, pulls a model or boots UE. Stdlib only.

Settings: opt-in, base rate 0.05 (a constant; no poll sets it), pool held,
mint false, socket false, fetch false, layer_one false, ue5_launch false,
bind 127.0.0.1 only. Each link closes with sha256(prev || pack || text)[:16].
T13 (juniorctl skill-pin height) is owned by the overnight job; not touched.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
OPT_IN = True
POOL_HELD = True
MINT = False
SOCKET = False
FETCH = False
LAYER_ONE = False
UE5_LAUNCH = False
MODEL_PULL = False
HOST = "127.0.0.1"
BITNETD_PORT = 8765
I2SD_PORT = 8767
ORDER = (("bitnetd", BITNETD_PORT), ("i2sd", I2SD_PORT))  # fixed, never reversed
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE_RULE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ITEM = "JuniorOS overlay (BETA_TO_OS) — bitnetd 127.0.0.1:8765 -> i2sd 127.0.0.1:8767 fallback"

# What each port contributes to the handover, from Home's point of view.
_ROLES = {
    "Home": "reads handover order; picks lane; never opens it",
    "OSai": "agent layer; asks the chosen lane only",
    "FieldCore": "jsonl intent; records lane + reason",
    "Flagstaff": "AND gate; denies dual-active or reversed order",
    "JuniorLLM": "overlay doc BETA_TO_OS; bitnetd.service owner",
}

# One balanced-trit pack per lane outcome: 1 = chosen, 0 = held, -1 = denied.
_ENC = {0: 0, 1: 1, -1: 2}

# Static default facts: nothing ready, nothing run. Handover holds.
DEFAULT_FACTS = {
    "opt_in": True,
    "bitnetd": {"ready": False, "bind": f"{HOST}:{BITNETD_PORT}"},
    "i2sd": {"ready": False, "bind": f"{HOST}:{I2SD_PORT}"},
}


def pack(trits) -> bytes:
    """Balanced trits as 2-bit codes (0->0, 1->1, -1->2), MSB first, zero-padded."""
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


def close(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def _bind_ok(bind, port: int) -> bool:
    return bind == f"{HOST}:{port}"


def gate(facts: dict) -> list:
    """Reasons to deny the whole handover. Empty list = may proceed."""
    bad = []
    if not facts.get("opt_in"):
        bad.append("opt_in")
    for name, port in ORDER:
        lane = facts.get(name) or {}
        if "bind" in lane and not _bind_ok(lane["bind"], port):
            bad.append(f"{name}.bind")
        if lane.get("active") and lane.get("ready") is False:
            bad.append(f"{name}.active_not_ready")
    if all((facts.get(n) or {}).get("active") for n, _ in ORDER):
        bad.append("dual_active")
    if facts.get("order") and tuple(facts["order"]) != tuple(n for n, _ in ORDER):
        bad.append("order")
    for k in ("socket", "fetch", "mint", "layer_one", "ue5_launch",
              "model_pull", "rate_poll", "pool_release"):
        if facts.get(k):
            bad.append(k)
    if "rate" in facts and facts["rate"] != RATE:
        bad.append("rate")
    return bad


def choose(facts: dict) -> tuple:
    """Return (lane, port, trits, deny). bitnetd first, then i2sd, else hold."""
    bad = gate(facts)
    if bad:
        return None, None, (-1, -1), bad
    trits = []
    picked = (None, None)
    for name, port in ORDER:
        ready = bool((facts.get(name) or {}).get("ready"))
        if ready and picked[0] is None:
            picked = (name, port)
            trits.append(1)
        else:
            trits.append(0)
    return picked[0], picked[1], tuple(trits), []


def handshake(facts: dict | None = None) -> dict:
    f = dict(DEFAULT_FACTS if facts is None else facts)
    lane, port, trits, bad = choose(f)
    pk = pack(trits)
    prev = GENESIS
    links = []
    for p in PORTS:
        text = json.dumps({"port": p, "role": _ROLES[p], "lane": lane,
                           "lane_port": port, "rate": RATE}, sort_keys=True)
        prev = close(prev, pk, text)
        links.append({"port": p, "close": prev})
    return {
        "item": ITEM, "opt_in": OPT_IN, "rate": RATE, "pool_held": POOL_HELD,
        "mint": MINT, "socket": SOCKET, "fetch": FETCH, "layer_one": LAYER_ONE,
        "ue5_launch": UE5_LAUNCH, "host": HOST,
        "order": [f"{n}:{HOST}:{p}" for n, p in ORDER],
        "lane": lane, "lane_endpoint": f"http://{HOST}:{port}" if port else None,
        "state": "denied" if bad else ("chosen" if lane else "held"),
        "deny": bad, "trits": list(trits), "links": links, "close": prev,
    }


def selftest() -> int:
    fails = []
    ran = []

    def check(name, cond):
        ran.append(name)
        if not cond:
            fails.append(name)

    check("rate", RATE == 0.05)
    check("flags", OPT_IN and POOL_HELD and not (MINT or SOCKET or FETCH
          or LAYER_ONE or UE5_LAUNCH or MODEL_PULL))
    check("host", HOST == "127.0.0.1")
    check("order", ORDER == (("bitnetd", 8765), ("i2sd", 8767)))
    check("pack", pack((1, 0, -1, 1)) == bytes([0b01001001]) and pack((1,)) == b"\x40")
    check("close", close(GENESIS, b"\x01", "x")
          == hashlib.sha256(GENESIS.encode() + b"\x01" + b"x").hexdigest()[:16])

    h = handshake()
    check("default_held", h["state"] == "held" and h["lane"] is None)
    check("links", [l["port"] for l in h["links"]] == list(PORTS))
    # re-derive chain independently
    prev = GENESIS
    pk = pack(tuple(h["trits"]))
    for l in h["links"]:
        text = json.dumps({"port": l["port"], "role": _ROLES[l["port"]], "lane": None,
                           "lane_port": None, "rate": RATE}, sort_keys=True)
        prev = hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]
        check(f"chain.{l['port']}", prev == l["close"])
    check("deterministic", handshake()["close"] == h["close"])

    both = {"opt_in": True, "bitnetd": {"ready": True, "bind": "127.0.0.1:8765"},
            "i2sd": {"ready": True, "bind": "127.0.0.1:8767"}}
    hb = handshake(both)
    check("primary_bitnetd", hb["lane"] == "bitnetd" and hb["lane_endpoint"] == "http://127.0.0.1:8765")
    check("primary_trits", hb["trits"] == [1, 0])

    fb = {"opt_in": True, "bitnetd": {"ready": False, "bind": "127.0.0.1:8765"},
          "i2sd": {"ready": True, "bind": "127.0.0.1:8767"}}
    hf = handshake(fb)
    check("fallback_i2sd", hf["lane"] == "i2sd" and hf["lane_endpoint"] == "http://127.0.0.1:8767")
    check("fallback_trits", hf["trits"] == [0, 1])
    check("close_differs", len({h["close"], hb["close"], hf["close"]}) == 3)

    denials = [
        ("no_opt_in", {**both, "opt_in": False}, "opt_in"),
        ("wide_bitnetd", {**both, "bitnetd": {"ready": True, "bind": "0.0.0.0:8765"}}, "bitnetd.bind"),
        ("wrong_i2sd", {**both, "i2sd": {"ready": True, "bind": "127.0.0.1:8768"}}, "i2sd.bind"),
        ("dual_active", {**both, "bitnetd": {"ready": True, "active": True, "bind": "127.0.0.1:8765"},
                         "i2sd": {"ready": True, "active": True, "bind": "127.0.0.1:8767"}}, "dual_active"),
        ("reversed", {**both, "order": ["i2sd", "bitnetd"]}, "order"),
        ("rate_poll", {**both, "rate_poll": True}, "rate_poll"),
        ("rate_change", {**both, "rate": 0.07}, "rate"),
        ("socket", {**both, "socket": True}, "socket"),
        ("fetch", {**both, "fetch": True}, "fetch"),
        ("mint", {**both, "mint": True}, "mint"),
        ("layer_one", {**both, "layer_one": True}, "layer_one"),
        ("ue5", {**both, "ue5_launch": True}, "ue5_launch"),
        ("pool", {**both, "pool_release": True}, "pool_release"),
    ]
    for name, facts, reason in denials:
        d = handshake(facts)
        check(f"deny.{name}", d["state"] == "denied" and reason in d["deny"] and d["lane"] is None)

    if fails:
        print("selftest FAIL:", ", ".join(fails))
        return 1
    print(f"selftest ok: {len(ran)} checks, close {h['close']}")
    return 0


def main(argv) -> int:
    if "--selftest" in argv:
        return selftest()
    print(json.dumps(handshake(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
