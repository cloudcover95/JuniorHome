"""i2sd userspace privilege + loopback port ledger handshake.

ROADMAP item (docs/ROADMAP.md, Order #3, verbatim):
    **i2sd userspace** — 127.0.0.1:8767. No docker.sock.

New angle: the existing i2sd handshake checks bind and a docker_sock flag.
This file advances the same item one step further by pinning what
"userspace" means as a declarative privilege ledger for the i2sd lane:
non-root uid/gid, unprivileged port (>= 1024), no ambient capabilities
(no CAP_NET_BIND_SERVICE, no CAP_SYS_ADMIN), no docker.sock path in any
mount or env value, and a loopback port ledger in which bitnetd
127.0.0.1:8765, deck 127.0.0.1:8766 and i2sd 127.0.0.1:8767 are each unique
and never collide. Nothing is opened, bound, stat'ed or executed.

Ports touched (in-process handshake / port map only):
    Home, OSai, FieldCore, Flagstaff, JuniorLLM
Loopback lanes named (never opened):
    bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767 (deck 127.0.0.1:8766 ledger-only)

Existing path: opt-in true, AntiCon, one trit pack, base rate 0.05 (a
constant; no poll sets it), pool held, mint false, socket false, fetch false,
layer_one false, ue5_launch false, bind 127.0.0.1 only. No model pull, no UE
boot, no subprocess, no exec of repo code, no network. Stdlib only.
Close rule: sha256(prev || pack || text)[:16], balanced trits packed as
2-bit codes (0->0, 1->1, -1->2), MSB first, zero-padded.

Usage: python3 ports/i2sd_userspace_privilege_ledger.py [--selftest]
Exit 0 on pass, 1 on fail.
"""
from __future__ import annotations

import hashlib
import json
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
DOCKER_SOCK = False
I2SD_LIVE = False
BIND = "127.0.0.1"
BITNETD = (BIND, 8765)
DECK = (BIND, 8766)
I2SD = (BIND, 8767)
LEDGER = {"bitnetd": BITNETD, "deck": DECK, "i2sd": I2SD}
UNPRIVILEGED_MIN = 1024
DENY_CAPS = ("CAP_NET_BIND_SERVICE", "CAP_SYS_ADMIN", "CAP_NET_RAW", "CAP_NET_ADMIN")
DOCKER_SOCK_PATHS = ("docker.sock", "/var/run/docker.sock", "/run/docker.sock")
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "**i2sd userspace** — 127.0.0.1:8767. No docker.sock."
ROADMAP_LINE = "3. " + ROADMAP_ITEM
NAME = "i2sd-userspace-privilege-ledger"
DAY = "2026-10-18"
ORIGIN = "operator"

_ROLES = {
    "Home": "operator box / declares i2sd unit spec",
    "OSai": "agent layer / reads ledger",
    "FieldCore": "jsonl intent / ledger row",
    "Flagstaff": "6-vote AND / privilege gate",
    "JuniorLLM": "i2s consumer / bitnetd 8765 primary, i2sd 8767 lane",
}
# ONE trit pack for the whole handshake (balanced -1/0/1), reused every link.
TRIT_PACK = (1, -1, 0, 1, 1, 0, -1, 1)
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}

# Static declared spec (what the operator intends i2sd to run as). Callers may
# inject their own dict; nothing is collected live.
DEFAULT_SPEC = {
    "origin": ORIGIN,
    "uid": 1000,
    "gid": 1000,
    "bind": BIND,
    "port": 8767,
    "caps": [],
    "mounts": ["/home/junior/models:ro"],
    "env": {"JUNIOR_I2SD_BIND": "127.0.0.1:8767"},
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


def ledger_check(ledger: dict = LEDGER) -> dict:
    """Every lane loopback-only and unique; i2sd is 8767, bitnetd is 8765."""
    bad = []
    seen = {}
    for name, (host, port) in ledger.items():
        if host != BIND:
            bad.append(f"{name}:bind")
        if port in seen:
            bad.append(f"{name}:collides:{seen[port]}")
        seen[port] = name
    if ledger.get("i2sd") != I2SD:
        bad.append("i2sd:port")
    if ledger.get("bitnetd") != BITNETD:
        bad.append("bitnetd:port")
    return {"ok": not bad, "reject": bad}


def _mentions_docker_sock(values) -> bool:
    return any(p in str(v) for v in values for p in DOCKER_SOCK_PATHS)


def anticon(spec: dict) -> dict:
    """AntiCon privilege gate: reject anything that is not plain userspace."""
    bad = []
    if spec.get("origin") != ORIGIN:
        bad.append("origin")
    if spec.get("bind") != BIND:
        bad.append("bind")
    if spec.get("port") != I2SD[1]:
        bad.append("port")
    if int(spec.get("port", 0)) < UNPRIVILEGED_MIN:
        bad.append("privileged_port")
    if int(spec.get("uid", 0)) == 0 or int(spec.get("gid", 0)) == 0:
        bad.append("root")
    if any(c in DENY_CAPS for c in spec.get("caps", [])):
        bad.append("caps")
    if _mentions_docker_sock(spec.get("mounts", [])) or _mentions_docker_sock(spec.get("env", {}).values()):
        bad.append("docker_sock")
    for k in ("socket", "fetch", "mint", "layer_one", "ue5_launch",
              "model_pull", "docker_sock", "rate_poll", "live"):
        if spec.get(k):
            bad.append(k)
    if "rate" in spec and spec["rate"] != RATE:
        bad.append("rate")
    return {"ok": not bad, "reject": bad}


def port_map() -> dict:
    """Conceptual in-process port map (never opened)."""
    return {p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET, "fetch": FETCH} for p in PORTS}


def _text(port: str, ac: dict, lg: dict) -> str:
    return (
        f"{NAME}:{DAY}:{port}:{_ROLES[port]}:item={ROADMAP_ITEM}:"
        f"anticon={ac['ok']}:ledger={lg['ok']}:opt_in={OPT_IN}:pool_held={POOL_HELD}:"
        f"mint={MINT}:socket={SOCKET}:fetch={FETCH}:layer_one={LAYER_ONE}:"
        f"ue5_launch={UE5_LAUNCH}:docker_sock={DOCKER_SOCK}:live={I2SD_LIVE}:"
        f"bitnetd={BITNETD[0]}:{BITNETD[1]}:i2sd={I2SD[0]}:{I2SD[1]}:bind={BIND}:rate={RATE}"
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


def handshake(spec: dict | None = None, ledger: dict = LEDGER) -> dict:
    s = dict(DEFAULT_SPEC if spec is None else spec)
    ac = anticon(s)
    lg = ledger_check(ledger)
    chain = build_chain(_text(p, ac, lg) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": bool(v["ok"] and ac["ok"] and lg["ok"] and OPT_IN and len(chain) == len(PORTS)),
        "name": NAME,
        "roadmap_item": ROADMAP_ITEM,
        "ports_touched": list(PORTS),
        "lanes": {"bitnetd": f"{BITNETD[0]}:{BITNETD[1]}", "i2sd": f"{I2SD[0]}:{I2SD[1]}"},
        "anticon": ac,
        "ledger": lg,
        "i2sd_live": I2SD_LIVE,
        "close_rule": CLOSE,
        "close": v.get("tip"),
        "chain": chain,
    }


def selftest() -> dict:
    assert RATE == 0.05 and OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or FETCH or LAYER_ONE or UE5_LAUNCH or MODEL_PULL or DOCKER_SOCK or I2SD_LIVE)
    assert BIND == "127.0.0.1" and BITNETD == ("127.0.0.1", 8765) and I2SD == ("127.0.0.1", 8767)
    assert "i2sd userspace" in ROADMAP_ITEM and "No docker.sock." in ROADMAP_ITEM
    for mod in ("socket", "subprocess", "urllib.request", "http.client"):
        assert mod not in sys.modules or mod == "socket" and SOCKET is False
    assert unpack(PACK, len(TRIT_PACK)) == list(TRIT_PACK)
    assert close(GENESIS, PACK, "x") == hashlib.sha256(GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    for v in port_map().values():
        assert v["bind"] == BIND and v["socket"] is False and v["fetch"] is False
    hs = handshake()
    assert hs["ok"] and hs["ports_touched"] == list(PORTS) and len(hs["close"]) == 16
    assert all(f"rate={RATE}" in r["text"] and "pool_held=True" in r["text"] for r in hs["chain"])
    # privilege widening is rejected
    for k, v in (("uid", 0), ("gid", 0), ("port", 80), ("port", 8765), ("bind", "0.0.0.0"),
                 ("caps", ["CAP_NET_BIND_SERVICE"]), ("mounts", ["/var/run/docker.sock:/var/run/docker.sock"]),
                 ("env", {"DOCKER_HOST": "unix:///run/docker.sock"}), ("origin", "bot"),
                 ("socket", True), ("fetch", True), ("mint", True), ("model_pull", True),
                 ("ue5_launch", True), ("rate_poll", True), ("rate", 0.5), ("live", True)):
        assert handshake(dict(DEFAULT_SPEC, **{k: v}))["ok"] is False, (k, v)
    # ledger collisions / non-loopback lanes are rejected
    assert handshake(ledger=dict(LEDGER, deck=(BIND, 8767)))["ok"] is False
    assert handshake(ledger=dict(LEDGER, i2sd=("0.0.0.0", 8767)))["ok"] is False
    assert handshake(ledger=dict(LEDGER, bitnetd=(BIND, 8768)))["ok"] is False
    # tamper detection
    bad = [dict(r) for r in hs["chain"]]
    bad[2]["text"] = bad[2]["text"].replace("FieldCore", "FieldC0re")
    assert verify_chain(bad) == {"ok": False, "at": 2, "reason": "commit"}
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
    out = {k: hs[k] for k in ("ok", "name", "roadmap_item", "ports_touched", "lanes",
                              "anticon", "ledger", "i2sd_live", "close_rule", "close")}
    out["constants"] = {"opt_in": OPT_IN, "anticon": ANTICON, "rate": RATE, "pool_held": POOL_HELD,
                        "mint": MINT, "socket": SOCKET, "fetch": FETCH, "layer_one": LAYER_ONE,
                        "ue5_launch": UE5_LAUNCH, "bind": BIND, "trit_pack": list(TRIT_PACK)}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if hs["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
