"""GGUF download-deny provenance ledger handshake.

ROADMAP item (docs/ROADMAP.md, Order #2, verbatim):
    **GGUF** — operator sets JUNIOR_GGUF. Bots do not download. `web3node/probe_future.py`.

New angle: the existing GGUF handshakes check that JUNIOR_GGUF is set and
that the file starts with the "GGUF" magic. This file pins the other half of
the item, "Bots do not download", as a declarative provenance ledger for
the JUNIOR_GGUF *value*: it must be set by the operator (never a bot), must
be a plain local absolute path ending in .gguf, and must not be anything a
download would produce or need: no URL scheme (http, https, hf, s3, ftp,
git+...), no hub repo id ("org/model"), no download-cache location
(~/.cache/huggingface, hub/, lm-studio, ollama blobs, /tmp), no partial-file
suffix (.part, .incomplete, .tmp, .download, .crdownload), no "..". An unset
value holds (not a failure); a clean value maps to the bitnetd consumer on
127.0.0.1:8765 with i2sd 127.0.0.1:8767 as the second lane. String logic
only: the path is never stat'ed or opened, the env is never read live, and
web3node/probe_future.py is named as the operator's reference only; it is
not imported, run or wrapped.

Ports touched (in-process handshake / port map only):
    Home, OSai, FieldCore, Flagstaff, JuniorLLM
Loopback lanes named (never opened):
    bitnetd 127.0.0.1:8765 (primary), i2sd 127.0.0.1:8767 (second)

Existing path: opt-in true, AntiCon, one trit pack, base rate 0.05 (a
constant; no poll sets it), pool held, mint false, socket false, fetch false,
layer_one false, ue5_launch false, bind 127.0.0.1 only. No model pull, no
download, no UE boot, no subprocess, no exec of repo code, no network.
Close rule: sha256(prev || pack || text)[:16], balanced trits packed as
2-bit codes (0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.

Usage: python3 ports/gguf_download_deny_ledger.py [--selftest]
Exit 0 on pass, 1 on fail.
"""
from __future__ import annotations

import hashlib
import json
import re
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
DOWNLOAD = False
BOT_SETS_GGUF = False
BIND = "127.0.0.1"
BITNETD = (BIND, 8765)
I2SD = (BIND, 8767)
ENV_KEY = "JUNIOR_GGUF"
REFERENCE = "web3node/probe_future.py"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "**GGUF** — operator sets JUNIOR_GGUF. Bots do not download. `web3node/probe_future.py`."
ROADMAP_LINE = "2. " + ROADMAP_ITEM
NAME = "gguf-download-deny-ledger"
DAY = "2026-10-19"
ORIGIN = "operator"

SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
CACHE_MARKERS = ("/.cache/huggingface", "/huggingface/hub", "/hub/models--", "/.cache/lm-studio",
                 "/.lmstudio/", "/.ollama/models/blobs", "/tmp/", "/var/tmp/")
PARTIAL_SUFFIXES = (".part", ".incomplete", ".tmp", ".download", ".crdownload", ".partial")

_ROLES = {
    "Home": "operator box / sets JUNIOR_GGUF",
    "OSai": "agent layer / reads value, never sets it",
    "FieldCore": "jsonl intent / provenance row",
    "Flagstaff": "6-vote AND / download-deny gate",
    "JuniorLLM": "bitnetd 8765 consumer, i2sd 8767 second lane",
}
# ONE trit pack for the whole handshake (balanced -1/0/1), reused every link.
TRIT_PACK = (0, 1, -1, 1, -1, 0, 1, 1)
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}

# Static declared facts. Default: JUNIOR_GGUF unset, so the lane holds.
# Callers may inject their own dict; the env is never read live.
DEFAULT_FACTS = {"origin": ORIGIN, "setter": ORIGIN, "junior_gguf": None}


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


def value_check(value) -> dict:
    """Pure string provenance check of a JUNIOR_GGUF value. Never stats or opens it."""
    if value is None or value == "":
        return {"state": "held", "ok": True, "reject": []}
    v = str(value)
    bad = []
    if SCHEME_RE.match(v) or "://" in v:
        bad.append("scheme")
    if REPO_ID_RE.match(v):
        bad.append("repo_id")
    if not v.startswith("/"):
        bad.append("not_absolute")
    if ".." in v.split("/"):
        bad.append("dotdot")
    low = v.lower()
    if any(m in low for m in CACHE_MARKERS):
        bad.append("download_cache")
    if any(low.endswith(s) for s in PARTIAL_SUFFIXES):
        bad.append("partial")
    elif not low.endswith(".gguf"):
        bad.append("not_gguf")
    return {"state": "reject" if bad else "ready", "ok": not bad, "reject": bad}


def anticon(facts: dict) -> dict:
    """AntiCon gate: only the operator sets JUNIOR_GGUF; nothing downloads."""
    bad = []
    if facts.get("origin") != ORIGIN:
        bad.append("origin")
    if facts.get("setter") != ORIGIN:
        bad.append("setter")
    if "bind" in facts and facts["bind"] != BIND:
        bad.append("bind")
    for k in ("socket", "fetch", "mint", "layer_one", "ue5_launch", "model_pull",
              "download", "bot_sets_gguf", "rate_poll", "probe_future_exec"):
        if facts.get(k):
            bad.append(k)
    if "rate" in facts and facts["rate"] != RATE:
        bad.append("rate")
    return {"ok": not bad, "reject": bad}


def lane(vc: dict) -> dict:
    """Map a clean value to the bitnetd consumer; i2sd is only the second lane."""
    ready = vc["state"] == "ready"
    return {
        "primary": f"{BITNETD[0]}:{BITNETD[1]}",
        "second": f"{I2SD[0]}:{I2SD[1]}",
        "route": "bitnetd" if ready else "held",
        "opened": False,
    }


def port_map() -> dict:
    """Conceptual in-process port map (never opened)."""
    return {p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET, "fetch": FETCH} for p in PORTS}


def _text(port: str, ac: dict, vc: dict, ln: dict) -> str:
    return (
        f"{NAME}:{DAY}:{port}:{_ROLES[port]}:item={ROADMAP_ITEM}:key={ENV_KEY}:"
        f"state={vc['state']}:route={ln['route']}:anticon={ac['ok']}:opt_in={OPT_IN}:"
        f"pool_held={POOL_HELD}:mint={MINT}:socket={SOCKET}:fetch={FETCH}:"
        f"layer_one={LAYER_ONE}:ue5_launch={UE5_LAUNCH}:download={DOWNLOAD}:"
        f"bitnetd={ln['primary']}:i2sd={ln['second']}:bind={BIND}:rate={RATE}"
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


def handshake(facts: dict | None = None) -> dict:
    f = dict(DEFAULT_FACTS if facts is None else facts)
    ac = anticon(f)
    vc = value_check(f.get("junior_gguf"))
    ln = lane(vc)
    chain = build_chain(_text(p, ac, vc, ln) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": bool(v["ok"] and ac["ok"] and vc["ok"] and OPT_IN and len(chain) == len(PORTS)),
        "name": NAME,
        "roadmap_item": ROADMAP_ITEM,
        "ports_touched": list(PORTS),
        "lanes": {"bitnetd": ln["primary"], "i2sd": ln["second"]},
        "env_key": ENV_KEY,
        "value_state": vc["state"],
        "route": ln["route"],
        "anticon": ac,
        "value": vc,
        "download": DOWNLOAD,
        "reference": REFERENCE,
        "close_rule": CLOSE,
        "close": v.get("tip"),
        "chain": chain,
    }


def selftest() -> dict:
    assert RATE == 0.05 and OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or FETCH or LAYER_ONE or UE5_LAUNCH or MODEL_PULL or DOWNLOAD or BOT_SETS_GGUF)
    assert BIND == "127.0.0.1" and BITNETD == ("127.0.0.1", 8765) and I2SD == ("127.0.0.1", 8767)
    assert "Bots do not download." in ROADMAP_ITEM and ENV_KEY in ROADMAP_ITEM and REFERENCE in ROADMAP_ITEM
    for mod in ("subprocess", "urllib.request", "http.client", "probe_future"):
        assert mod not in sys.modules, mod
    assert unpack(PACK, len(TRIT_PACK)) == list(TRIT_PACK)
    assert close(GENESIS, PACK, "x") == hashlib.sha256(GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    for v in port_map().values():
        assert v["bind"] == BIND and v["socket"] is False and v["fetch"] is False
    hs = handshake()
    assert hs["ok"] and hs["value_state"] == "held" and hs["route"] == "held"
    assert hs["ports_touched"] == list(PORTS) and len(hs["close"]) == 16
    assert all(f"rate={RATE}" in r["text"] and "pool_held=True" in r["text"] for r in hs["chain"])
    # a clean operator-set local path routes to bitnetd 8765 (still never opened)
    good = handshake(dict(DEFAULT_FACTS, junior_gguf="/srv/junior/models/bitnet-b1.58-2B-4T-i2_s.gguf"))
    assert good["ok"] and good["route"] == "bitnetd" and good["lanes"]["bitnetd"] == "127.0.0.1:8765"
    # anything a download would produce or need is rejected
    for val, why in (("https://huggingface.co/x/y/resolve/main/m.gguf", "scheme"),
                     ("hf://microsoft/bitnet-b1.58-2B-4T-gguf", "scheme"),
                     ("s3://bucket/m.gguf", "scheme"),
                     ("microsoft/bitnet-b1.58-2B-4T-gguf", "repo_id"),
                     ("models/m.gguf", "not_absolute"),
                     ("/srv/junior/../../tmp/m.gguf", "dotdot"),
                     ("/home/n/.cache/huggingface/hub/models--x/m.gguf", "download_cache"),
                     ("/home/n/.ollama/models/blobs/sha256-abc", "download_cache"),
                     ("/tmp/m.gguf", "download_cache"),
                     ("/srv/junior/models/m.gguf.part", "partial"),
                     ("/srv/junior/models/m.gguf.incomplete", "partial"),
                     ("/srv/junior/models/m.safetensors", "not_gguf")):
        r = handshake(dict(DEFAULT_FACTS, junior_gguf=val))
        assert r["ok"] is False and why in r["value"]["reject"], (val, r["value"])
    # bots never set the value, and nothing widens the path
    for k, v in (("setter", "bot"), ("origin", "automation"), ("download", True), ("model_pull", True),
                 ("bot_sets_gguf", True), ("probe_future_exec", True), ("socket", True), ("fetch", True),
                 ("mint", True), ("ue5_launch", True), ("rate_poll", True), ("rate", 0.5), ("bind", "0.0.0.0")):
        assert handshake(dict(DEFAULT_FACTS, **{k: v}))["ok"] is False, (k, v)
    # tamper detection
    bad = [dict(r) for r in hs["chain"]]
    bad[3]["text"] = bad[3]["text"].replace("Flagstaff", "Flagstaf")
    assert verify_chain(bad) == {"ok": False, "at": 3, "reason": "commit"}
    assert verify_chain(hs["chain"], pk=pack((0,) * len(TRIT_PACK)))["ok"] is False
    return {"selftest": "pass", "name": NAME, "roadmap_item": ROADMAP_ITEM,
            "ports_touched": list(PORTS), "lanes": hs["lanes"], "close": hs["close"]}


def main(argv) -> int:
    if "--selftest" in argv:
        try:
            print(json.dumps(selftest(), ensure_ascii=False))
            return 0
        except AssertionError as e:
            print(json.dumps({"selftest": "fail", "error": repr(e)}, ensure_ascii=False))
            return 1
    hs = handshake()
    out = {k: hs[k] for k in ("ok", "name", "roadmap_item", "ports_touched", "lanes", "env_key",
                              "value_state", "route", "anticon", "value", "download", "reference",
                              "close_rule", "close")}
    out["constants"] = {"opt_in": OPT_IN, "anticon": ANTICON, "rate": RATE, "pool_held": POOL_HELD,
                        "mint": MINT, "socket": SOCKET, "fetch": FETCH, "layer_one": LAYER_ONE,
                        "ue5_launch": UE5_LAUNCH, "bind": BIND, "trit_pack": list(TRIT_PACK)}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if hs["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
