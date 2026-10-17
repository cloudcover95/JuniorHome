"""JuniorOS overlay base handshake: Debian/Alpine split, juniorctl on PATH, loopback ports.

Advances JuniorLLM docs/BETA_TO_OS.md item "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH) from the
JuniorHome side. New angle: the base split. Debian runs bitnetd as a systemd
unit; Alpine has no systemd and runs it as an OpenRC service. This check takes
the host os-release text plus the overlay os-release.junior text (both passed
in as strings), decides the base, maps the init system and service paths per
base, resolves juniorctl from a PATH string (string logic only, no exec, no
stat), and lists the live-beta handshake order: juniorctl health, then
bitnetd 127.0.0.1:8765, then i2sd 127.0.0.1:8767. Ports are listed, never
opened. It does not import, wrap or re-check JuniorLLM ports/overlay_port_map.py
or any other team file.

No sockets, no network, no subprocess, no fetch, no download, no model pull.
Stdlib only. Close rule: sha256(prev || pack || text)[:16] with balanced trits
packed as 2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded. Rate 0.05.
"""
from __future__ import annotations

import hashlib
import os

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
DOWNLOAD = False
MODEL_PULL = False
EXEC = False
BIND = "127.0.0.1"
BITNETD_PORT = 8765
I2SD_PORT = 8767
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
DAY = "2026-10-16"
ROADMAP_ITEM = (
    "JuniorOS overlay — Debian/Alpine + os-release.junior + hardened "
    "bitnetd.service + juniorctl on PATH."
)
JUNIORCTL_DIR = "/usr/local/bin"
HANDSHAKE_ORDER = ("juniorctl health", "bitnetd", "i2sd")

BASES = {
    "debian": {
        "init": "systemd",
        "bitnetd": "/etc/systemd/system/bitnetd.service",
        "i2sd": "/etc/systemd/system/i2sd.service",
        "enable": "systemctl enable bitnetd.service",
    },
    "alpine": {
        "init": "openrc",
        "bitnetd": "/etc/init.d/bitnetd",
        "i2sd": "/etc/init.d/i2sd",
        "enable": "rc-update add bitnetd default",
    },
}

_ROLES = {
    "Home": "operator host, base os-release",
    "OSai": "reads os-release.junior overlay layer",
    "FieldCore": "jsonl intent, base-agnostic",
    "Flagstaff": "6-vote AND on overlay ok",
    "JuniorLLM": "bitnetd consumer on loopback",
}
_TRITS = {
    "Home": (1, -1, 0, 1),
    "OSai": (0, 1, 1, -1, 0),
    "FieldCore": (-1, 0, 1, 0),
    "Flagstaff": (1, 1, -1, 0, 0, 1),
    "JuniorLLM": (0, -1, 1, 1),
}
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}

SAMPLE_DEBIAN = 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nID=debian\nVERSION_ID="12"\n'
SAMPLE_ALPINE = 'NAME="Alpine Linux"\nID=alpine\nVERSION_ID=3.20.3\n'
SAMPLE_UBUNTU = 'NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\n'
SAMPLE_FEDORA = 'NAME="Fedora Linux"\nID=fedora\nVERSION_ID=40\n'
SAMPLE_JUNIOR = (
    'NAME="JuniorOS"\nID=junioros\nID_LIKE="debian alpine"\n'
    'VARIANT_ID=overlay\nJUNIOR_BITNETD="127.0.0.1:8765"\n'
    'JUNIOR_I2SD="127.0.0.1:8767"\n'
)


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


def parse_os_release(text: str) -> dict:
    """KEY=value lines, quotes stripped, comments and junk ignored."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k.replace("_", "").isalnum():
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def detect_base(host: dict):
    """debian or alpine from ID, then ID_LIKE; None if neither."""
    ids = [host.get("ID", "").lower()] + host.get("ID_LIKE", "").lower().split()
    for i in ids:
        if i in BASES:
            return i
    return None


def _loopback(value: str, port: int) -> bool:
    return value == f"{BIND}:{port}"


def check_junior(junior: dict) -> dict:
    like = junior.get("ID_LIKE", "").lower().split()
    checks = {
        "id": junior.get("ID") == "junioros",
        "like": "debian" in like and "alpine" in like,
        "variant": junior.get("VARIANT_ID") == "overlay",
        "bitnetd": _loopback(junior.get("JUNIOR_BITNETD", ""), BITNETD_PORT),
        "i2sd": _loopback(junior.get("JUNIOR_I2SD", ""), I2SD_PORT),
    }
    return {"ok": all(checks.values()), "checks": checks}


def resolve_juniorctl(path: str) -> dict:
    """String-only PATH order. No stat, no exec: reports where juniorctl would sit."""
    entries = [p.rstrip("/") or "/" for p in (path or "").split(":") if p]
    on_path = JUNIORCTL_DIR in entries
    shadowed_by = entries[: entries.index(JUNIORCTL_DIR)] if on_path else []
    return {
        "dir": JUNIORCTL_DIR,
        "on_path": on_path,
        "index": entries.index(JUNIORCTL_DIR) if on_path else None,
        "earlier": shadowed_by,
        "relative_entry": any(not p.startswith("/") for p in entries),
        "exec": EXEC,
    }


def overlay_map(host_text: str, junior_text: str, path: str) -> dict:
    host = parse_os_release(host_text)
    junior = parse_os_release(junior_text)
    base = detect_base(host)
    jr = check_junior(junior)
    ctl = resolve_juniorctl(path)
    svc = BASES.get(base)
    ok = bool(svc) and jr["ok"] and ctl["on_path"] and not ctl["relative_entry"]
    return {
        "ok": ok,
        "base": base,
        "host_id": host.get("ID"),
        "init": svc["init"] if svc else None,
        "services": {k: svc[k] for k in ("bitnetd", "i2sd", "enable")} if svc else None,
        "junior": jr,
        "juniorctl": ctl,
        "order": list(HANDSHAKE_ORDER),
        "bitnetd": {"bind": BIND, "port": BITNETD_PORT, "socket": SOCKET},
        "i2sd": {"bind": BIND, "port": I2SD_PORT, "socket": SOCKET},
        "ports": {p: {"role": _ROLES[p], "bind": BIND, "socket": SOCKET} for p in PORTS},
    }


def _tag(port: str, m: dict) -> str:
    return (
        f"overlay-base:{DAY}:{port}:{_ROLES[port]}:base={m['base']}:init={m['init']}:"
        f"junior={m['junior']['ok']}:juniorctl={m['juniorctl']['on_path']}:"
        f"order={'>'.join(HANDSHAKE_ORDER)}:bind={BIND}:{BITNETD_PORT}:{I2SD_PORT}:"
        f"ok={m['ok']}:rate={RATE}"
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


def handshake_check(host_text: str, junior_text: str = SAMPLE_JUNIOR, path=None) -> dict:
    """One side-commit per named port over the overlay base map. In-process only."""
    path = os.environ.get("PATH", "") if path is None else path
    m = overlay_map(host_text, junior_text, path)
    chain = build_chain((_TRITS[p], _tag(p, m)) for p in PORTS)
    v = verify_chain(chain)
    return {
        "ok": m["ok"] and v["ok"] and len(chain) == len(PORTS),
        "roadmap_item": ROADMAP_ITEM,
        "map": m,
        "chain": chain,
        "tip": v.get("tip"),
        "close": CLOSE,
    }


def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert BITNETD_PORT == 8765 and I2SD_PORT == 8767
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert not (DOWNLOAD or MODEL_PULL or EXEC)
    assert RATE == 0.05
    assert unpack(pack([1, 0, -1, 1]), 4) == [1, 0, -1, 1]
    assert pack([1, 0, -1, 1]) == bytes([0x49])
    for p in PORTS:
        t = list(_TRITS[p])
        assert unpack(pack(t), len(t)) == t

    good = "/usr/local/sbin:/usr/local/bin:/usr/bin:/bin"
    deb = handshake_check(SAMPLE_DEBIAN, path=good)
    alp = handshake_check(SAMPLE_ALPINE, path=good)
    ubu = handshake_check(SAMPLE_UBUNTU, path=good)
    assert deb["ok"] and deb["map"]["init"] == "systemd"
    assert deb["map"]["services"]["bitnetd"].endswith("bitnetd.service")
    assert alp["ok"] and alp["map"]["init"] == "openrc"
    assert alp["map"]["services"]["bitnetd"] == "/etc/init.d/bitnetd"
    assert ubu["ok"] and ubu["map"]["base"] == "debian"
    assert deb["tip"] != alp["tip"]

    assert not handshake_check(SAMPLE_FEDORA, path=good)["ok"]
    assert not handshake_check(SAMPLE_DEBIAN, path="/usr/bin:/bin")["ok"]
    assert not handshake_check(SAMPLE_DEBIAN, path="bin:/usr/local/bin")["ok"]
    assert handshake_check(SAMPLE_DEBIAN, path="/usr/local/bin/:/bin")["ok"]
    wide = SAMPLE_JUNIOR.replace("127.0.0.1:8765", "0.0.0.0:8765")
    assert not handshake_check(SAMPLE_DEBIAN, wide, good)["ok"]
    wrong = SAMPLE_JUNIOR.replace("127.0.0.1:8767", "127.0.0.1:8768")
    assert not handshake_check(SAMPLE_ALPINE, wrong, good)["ok"]
    assert not handshake_check(SAMPLE_DEBIAN, "", good)["ok"]
    assert resolve_juniorctl(good)["earlier"] == ["/usr/local/sbin"]

    for hs in (deb, alp):
        assert hs["map"]["order"] == ["juniorctl health", "bitnetd", "i2sd"]
        assert all("0.0.0.0" not in r["text"] for r in hs["chain"])
        assert all(f"rate={RATE}" in r["text"] for r in hs["chain"])
        assert len(hs["tip"]) == 16 and verify_chain(hs["chain"])["ok"]

    t = [dict(r) for r in deb["chain"]]
    t[2]["text"] = t[2]["text"].replace("init=systemd", "init=openrc")
    assert verify_chain(t) == {"ok": False, "at": 2, "reason": "commit"}
    t = [dict(r) for r in deb["chain"]]
    t[4]["trits"] = [0] * len(t[4]["trits"])
    assert verify_chain(t) == {"ok": False, "at": 4, "reason": "commit"}
    t = [dict(r) for r in deb["chain"]]
    t[1]["prev"] = GENESIS
    assert verify_chain(t) == {"ok": False, "at": 1, "reason": "prev"}

    return {
        "ok": True,
        "roadmap_item": ROADMAP_ITEM,
        "day": DAY,
        "rate": RATE,
        "close": CLOSE,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "i2sd": f"{BIND}:{I2SD_PORT}",
        "debian_tip": deb["tip"],
        "alpine_tip": alp["tip"],
    }


if __name__ == "__main__":
    print(self_check())
