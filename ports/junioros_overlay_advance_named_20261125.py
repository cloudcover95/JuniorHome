"""JuniorOS overlay advance — ROADMAP / BETA_TO_OS named item.

Advances JuniorHome docs/ROADMAP.md and JuniorLLM docs/BETA_TO_OS.md named item:
JuniorOS overlay. (bitnetd 127.0.0.1:8765 / i2sd 127.0.0.1:8767 / juniorctl).
Pure in-process. Declarative facts only.
Never opens a socket, never pulls a model, never fetches, never boots UE,
never imports juniorctl or bitnetd.

Base rate 0.05. Pool held. mint false. socket false. fetch false.
layer_one false. ue5_launch false. Bind 127.0.0.1 only.
Close rule: sha256(prev || pack || text)[:16]. Stdlib only.
No wrap of path.py, chain_record.py, or record_block.py.
Additive only. Cap 1 file. Dated 20261125.
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
BITNETD = "127.0.0.1:8765"
I2SD = "127.0.0.1:8767"
JUNIORCTL = "juniorctl"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "JuniorOS overlay. (bitnetd 127.0.0.1:8765 / i2sd 127.0.0.1:8767 / juniorctl)"
DAY = "2026-11-25"
PROBE_ORIGIN = "operator"

_ROLES = {
    "Home": "operator box / JuniorOS overlay",
    "OSai": "agent layer / overlay reader",
    "FieldCore": "jsonl intent / overlay record",
    "Flagstaff": "6-vote AND / overlay gate",
    "JuniorLLM": "goldens / overlay suite",
}
TRIT_PACK = (1, 0, -1, 1, 0, -1, 1, 1)

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}

DEFAULT_FACTS = {
    "origin": PROBE_ORIGIN,
    "overlay": True,
    "bitnetd_bind": BITNETD,
    "i2sd_bind": I2SD,
    "juniorctl": JUNIORCTL,
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
              "model_pull", "rate_poll"):
        if facts.get(k):
            bad.append(k)
    if "rate" in facts and facts["rate"] != RATE:
        bad.append("rate")
    if facts.get("bitnetd_bind") and facts["bitnetd_bind"] != BITNETD:
        bad.append("bitnetd_bind")
    if facts.get("i2sd_bind") and facts["i2sd_bind"] != I2SD:
        bad.append("i2sd_bind")
    return {"ok": not bad, "reject": bad}


def probe(facts: dict | None = None) -> dict:
    """Declarative JuniorOS overlay probe. Reads injected facts only."""
    f = dict(DEFAULT_FACTS if facts is None else facts)
    ac = anticon(f)
    ready = bool(
        f.get("overlay")
        and f.get("bitnetd_bind") == BITNETD
        and f.get("i2sd_bind") == I2SD
        and f.get("juniorctl") == JUNIORCTL
    )
    return {
        "opt_in": OPT_IN,
        "anticon": ac,
        "counted": OPT_IN and ac["ok"],
        "origin": f.get("origin"),
        "overlay": bool(f.get("overlay")),
        "overlay_ready": ready and ac["ok"],
        "bitnetd": BITNETD,
        "i2sd": I2SD,
        "juniorctl": JUNIORCTL,
        "model_pull": MODEL_PULL,
    }


def port_map() -> dict:
    """Conceptual loopback port map (never opened)."""
    return {
        p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET, "fetch": FETCH}
        for p in PORTS
    }


def _text(port: str, pr: dict) -> str:
    return (
        f"junioros-overlay:{DAY}:{port}:{_ROLES[port]}:origin={pr['origin']}:"
        f"counted={pr['counted']}:opt_in={OPT_IN}:anticon={pr['anticon']['ok']}:"
        f"pool_held={POOL_HELD}:mint={MINT}:socket={SOCKET}:fetch={FETCH}:"
        f"layer_one={LAYER_ONE}:ue5_launch={UE5_LAUNCH}:bind={BIND}:rate={RATE}:"
        f"bitnetd={BITNETD}:i2sd={I2SD}:juniorctl={JUNIORCTL}:roadmap={ROADMAP_ITEM}"
    )


def build_chain(texts, prev: str = GENESIS) -> list:
    out = []
    for text in texts:
        c = close(prev, PACK, text)
        out.append({"prev": prev, "text": text, "commit": c})
        prev = c
    return out


def verify_chain(chain, pk: bytes = PACK) -> dict:
    prev = GENESIS
    for i, rec in enumerate(chain):
        if rec["prev"] != prev:
            return {"ok": False, "at": i, "reason": "prev"}
        if close(prev, pk, rec["text"]) != rec["commit"]:
            return {"ok": False, "at": i, "reason": "commit"}
        prev = rec["commit"]
    return {"ok": True, "tip": prev}


def handshake(facts: dict | None = None) -> dict:
    pr = probe(facts)
    texts = [_text(p, pr) for p in PORTS]
    chain = build_chain(texts)
    return {
        "probe": pr,
        "ports": port_map(),
        "chain": chain,
        "tip": chain[-1]["commit"] if chain else GENESIS,
    }


def handshake_check(facts: dict | None = None) -> dict:
    hs = handshake(facts)
    v = verify_chain(hs["chain"])
    return {"ok": v["ok"] and hs["probe"]["overlay_ready"], "verify": v, "handshake": hs}


def self_check() -> dict:
    hs = handshake()
    assert hs["probe"]["overlay_ready"] is True
    assert hs["probe"]["anticon"]["ok"] is True
    assert verify_chain(hs["chain"])["ok"] is True
    assert handshake_check()["ok"] is True
    # reject wideners
    for k, v in (("socket", True), ("fetch", True), ("mint", True),
                 ("model_pull", True), ("ue5_launch", True), ("rate_poll", True),
                 ("rate", 0.5), ("bind", "0.0.0.0"),
                 ("bitnetd_bind", "0.0.0.0:8765")):
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
        "bitnetd": BITNETD,
        "i2sd": I2SD,
        "juniorctl": JUNIORCTL,
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
    parser = argparse.ArgumentParser(description="JuniorOS overlay advance named")
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
