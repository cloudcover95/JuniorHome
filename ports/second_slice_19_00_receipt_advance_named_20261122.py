"""19:00 second slice if receipt >45 min old advance — ROADMAP named item.

Advances JuniorHome docs/ROADMAP.md named item:
Automations (current) ON: 19:00 — second slice if receipt >45 min old.
Pure in-process. Declarative facts only.
Never opens a socket, never pulls a model, never fetches, never boots UE,
never imports juniorctl or asahi.
Base rate 0.05. Pool held. mint false. socket false. fetch false.
layer_one false. ue5_launch false. Bind 127.0.0.1 only.
Close rule: sha256(prev || pack || text)[:16]. Stdlib only.
No wrap of path.py, chain_record.py, or record_block.py.
"""
from __future__ import annotations

import argparse
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
BIND = "127.0.0.1"
SECOND_SLICE = "19:00 if receipt >45 min old"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "Automations ON: 19:00 — second slice if receipt >45 min old."
DAY = "2026-11-22"
RECEIPT_AGE_MIN = 45

_ROLES = {
    "Home": "operator box / second slice receipt gate",
    "OSai": "agent layer / second slice reader",
    "FieldCore": "jsonl intent / second slice record",
    "Flagstaff": "6-vote AND / second slice gate",
    "JuniorLLM": "goldens / second slice suite",
}
TRIT_PACK = (1, 0, -1, 1, 0, -1, 1, 1)

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}

DEFAULT_FACTS = {
    "origin": "automations",
    "second_slice_enabled": True,
    "receipt_age_gt_45": True,
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
    if facts.get("origin") != "automations":
        bad.append("origin")
    for k in ("bind",):
        if k in facts and facts[k] != BIND:
            bad.append("bind")
    for k in ("socket", "fetch", "mint", "layer_one", "ue5_launch",
              "model_pull", "rate_poll"):
        if facts.get(k):
            bad.append(k)
    if "rate" in facts and facts["rate"] != RATE:
        bad.append("rate")
    if facts.get("second_slice_enabled") is False:
        bad.append("second_slice_enabled")
    if facts.get("receipt_age_gt_45") is False:
        bad.append("receipt_age_gt_45")
    return {"ok": not bad, "reject": bad}


def probe(facts: dict | None = None) -> dict:
    """Declarative 19:00 second slice if receipt >45 min old. Reads injected facts only."""
    f = dict(DEFAULT_FACTS if facts is None else facts)
    ac = anticon(f)
    ready = bool(
        f.get("second_slice_enabled")
        and f.get("receipt_age_gt_45")
        and f.get("origin") == "automations"
    )
    return {
        "opt_in": OPT_IN,
        "anticon": ac,
        "counted": OPT_IN and ac["ok"],
        "origin": f.get("origin"),
        "second_slice_ready": ready and ac["ok"],
        "second_slice": SECOND_SLICE,
        "model_pull": MODEL_PULL,
        "receipt_age_min": RECEIPT_AGE_MIN,
    }


def port_map() -> dict:
    """Conceptual loopback port map (never opened)."""
    return {
        p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET, "fetch": FETCH}
        for p in PORTS
    }


def _text(port: str, pr: dict) -> str:
    return (
        f"second-slice:{DAY}:{port}:{_ROLES[port]}:origin={pr['origin']}:"
        f"counted={pr['counted']}:opt_in={OPT_IN}:anticon={pr['anticon']['ok']}:"
        f"pool_held={POOL_HELD}:mint={MINT}:socket={SOCKET}:fetch={FETCH}:"
        f"layer_one={LAYER_ONE}:ue5_launch={UE5_LAUNCH}:bind={BIND}:rate={RATE}:"
        f"second_slice={SECOND_SLICE}:roadmap={ROADMAP_ITEM}"
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
    assert not (MINT or SOCKET or FETCH or LAYER_ONE or UE5_LAUNCH)
    assert not MODEL_PULL
    assert BIND == "127.0.0.1"
    assert "19:00" in ROADMAP_ITEM and "second slice if receipt >45 min old" in ROADMAP_ITEM
    assert unpack(PACK, len(TRIT_PACK)) == list(TRIT_PACK)
    for p, v in port_map().items():
        assert v["bind"] == BIND and v["socket"] is False and v["fetch"] is False
    c0 = close(GENESIS, PACK, "x")
    assert c0 == hashlib.sha256(GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    hs = handshake_check()
    assert hs["ok"] and hs["ports"] == list(PORTS)
    assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
    assert all(f"bind={BIND}" in r["text"] for r in hs["chain"])
    assert all("pool_held=True" in r["text"] for r in hs["chain"])
    assert all(f"second_slice={SECOND_SLICE}" in r["text"] for r in hs["chain"])
    assert len(hs["tip"]) == 16
    # non-automations is not counted
    assert probe({"origin": "manual"})["counted"] is False
    # AntiCon rejects widening attempts
    for k, v in (("socket", True), ("fetch", True), ("mint", True),
                 ("model_pull", True), ("ue5_launch", True), ("rate_poll", True),
                 ("rate", 0.5), ("bind", "0.0.0.0"),
                 ("second_slice_enabled", False), ("receipt_age_gt_45", False)):
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
        "second_slice": SECOND_SLICE,
        "opt_in": OPT_IN,
        "anticon": ANTICON,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "fetch": FETCH,
        "layer_one": LAYER_ONE,
        "ue5_launch": UE5_LAUNCH,
        "ports": list(PORTS),
        "tip": hs["tip"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="19:00 second slice if receipt >45 min old advance named")
    parser.add_argument("--selftest", action="store_true", help="run self-check and exit")
    args = parser.parse_args()
    if args.selftest or len(sys.argv) == 1:
        try:
            result = self_check()
            print(result)
        except AssertionError as e:
            print({"ok": False, "error": repr(e)})
            sys.exit(1)
        sys.exit(0)
    else:
        print("use --selftest")
        sys.exit(1)
