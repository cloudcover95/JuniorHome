#!/usr/bin/env python3
"""JuniorOS overlay netinst PATH port map: juniorctl on PATH per Debian/Alpine install lane.

Advances JuniorLLM docs/BETA_TO_OS.md item "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH), the
"juniorctl on PATH plus Debian/Alpine" part, from the JuniorHome side.

New angle: the install lanes, not the host base. JuniorLLM ships a Debian
netinst lane (rails/linux/debian: packages.list, preseed.fragment,
stage_overlay.sh) that runs install-overlay.sh with a DEST prefix. This map
takes those sources as text (an embedded snapshot pinned to JuniorLLM tip
184a480, or the same files read from a local tree with --juniorllm DIR; no
fetch), computes where juniorctl, junior-bitnetd, the profile.d PATH pin and
bitnetd.service land for each DEST, and resolves juniorctl against the stock
Debian and Alpine login PATH strings. String logic only: no exec, no stat of
system paths, no import of juniorctl.

Port rows: Home (operator DEST choice) -> JuniorLLM (installer layout) ->
OSai (juniorctl health first) -> FieldCore (row record) -> Flagstaff (lane
verdict). Live-beta handshake order is listed, never opened: juniorctl
health, then bitnetd 127.0.0.1:8765, then i2sd 127.0.0.1:8767.

It does not import, wrap, test or re-check any other team file
(ports/overlay_base_handshake.py, ports/junioros_overlay_handshake.py, or
JuniorLLM ports/overlay_*). No sockets, no network, no subprocess, no
download, no model pull. Stdlib only. Bind 127.0.0.1. Rate 0.05.
Close rule: sha256(prev || pack || text)[:16]; trits pass=1, advisory=0,
fail=-1 packed 2-bit i2s (0->0, 1->1, -1->2), MSB first, zero-padded.
Exit 1 on any fail row, else 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
EXEC = False
BIND = "127.0.0.1"
WILDCARD = ".".join(["0"] * 4)  # detected, never bound
BITNETD = "127.0.0.1:8765"
I2SD = "127.0.0.1:8767"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
DAY = "2026-10-18"
ROADMAP_ITEM = "BETA_TO_OS: JuniorOS overlay (juniorctl on PATH plus Debian/Alpine)"
SNAPSHOT_TIP = "184a480"
HANDSHAKE_ORDER = ("juniorctl health", "bitnetd " + BITNETD, "i2sd " + I2SD)

# Stock login PATH strings (Debian /etc/profile non-root; Alpine /etc/profile).
STOCK_PATH = {
    "debian": "/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games",
    "alpine": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
# Directories each init system actually scans for units/services.
INIT_SEARCH = {
    "debian": ("/etc/systemd/system", "/usr/lib/systemd/system", "/lib/systemd/system"),
    "alpine": ("/etc/init.d",),
}
PROFILE_SOURCED = "/etc/profile.d"

# Snapshot of JuniorLLM rails/linux sources at SNAPSHOT_TIP (text only).
SNAP = {
    "debian/packages.list": "python3\npython3-venv\nsystemd\nca-certificates\ngit\n",
    "debian/preseed.fragment": (
        "# Drop into a Debian netinst late_command. Uses stock linux-image-*\n"
        "d-i pkgsel/include string python3 git systemd\n"
        "# late_command example:\n"
        "# cp -a /media/usb/junioros-overlay/. /opt/junioros/\n"
        "# systemctl enable bitnetd.service\n"
    ),
    "debian/stage_overlay.sh": (
        "#!/bin/sh\nset -eu\n"
        'HERE=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)\n'
        "OUT=${1:-/tmp/junioros-overlay}\n"
        'export DEST="$OUT"\n'
        'sh "$HERE/install-overlay.sh"\n'
        'cp "$HERE/i2sd.py" "$HERE/backend.py" "$HERE/bitnet_cpp.py" "$OUT/usr/lib/junioros/"\n'
        'tar -C "$OUT" -czf "$OUT.tar.gz" .\n'
    ),
    "install-overlay.sh.layout": (
        "etc/os-release.junior\nusr/lib/os-release.junior\n"
        "etc/systemd/system/bitnetd.service\nusr/local/bin/junior-bitnetd\n"
        "usr/local/bin/juniorctl\netc/profile.d/junioros.sh\n"
        "usr/lib/systemd/user/bitnetd.service\netc/junioros/path.env\n"
    ),
    "profile.d/junioros.sh": (
        "# JuniorOS overlay PATH pin — loopback bitnetd only\n"
        'JUNIOROS_BIN="/usr/local/bin"\n'
        'case ":$PATH:" in\n'
        '  *":$JUNIOROS_BIN:"*) ;;\n'
        '  *) PATH="$JUNIOROS_BIN:$PATH" ;;\n'
        "esac\nexport PATH\n"
        'export JUNIOR_BIND="${JUNIOR_BIND:-127.0.0.1:8765}"\n'
    ),
    "juniorctl.py.shebang": "#!/usr/bin/env python3\n",
    "bitnetd.service.exec": "ExecStart=/usr/local/bin/junior-bitnetd\nEnvironment=JUNIOR_BIND=127.0.0.1:8765\n",
    "i2sd.py.port": 'HOST = "127.0.0.1"\nPORT = 8767\n',
    "os-release.junior": 'ID=junioros\nID_LIKE="debian alpine"\nJUNIOR_BITNETD="127.0.0.1:8765"\n',
}


def load_tree(root: str) -> dict:
    """Read the same facts from a local JuniorLLM tree (local read only)."""
    lin = os.path.join(root, "rails", "linux")

    def rd(*p):
        with open(os.path.join(lin, *p), encoding="utf-8") as fh:
            return fh.read()

    inst = rd("install-overlay.sh")
    prof = inst.split("junioros.sh\" <<'EOF'\n", 1)[1].split("\nEOF", 1)[0] + "\n"
    layout = []
    for line in inst.splitlines():
        s = line.strip()
        if s.startswith("cp ") and '"$DEST/' in s:
            layout.append(s.rsplit('"$DEST/', 1)[1].rstrip('"'))
        elif s.startswith("cat > ") and '"$DEST/' in s:
            layout.append(s.split('"$DEST/', 1)[1].split('"', 1)[0])
    svc = "".join(l + "\n" for l in rd("bitnetd.service").splitlines()
                  if l.startswith(("ExecStart=", "Environment=JUNIOR_BIND")))
    i2 = "".join(l + "\n" for l in rd("i2sd.py").splitlines()
                 if l.startswith(("HOST =", "PORT =")))
    return {
        "debian/packages.list": rd("debian", "packages.list"),
        "debian/preseed.fragment": rd("debian", "preseed.fragment"),
        "debian/stage_overlay.sh": rd("debian", "stage_overlay.sh"),
        "install-overlay.sh.layout": "".join(p + "\n" for p in layout),
        "profile.d/junioros.sh": prof,
        "juniorctl.py.shebang": rd("juniorctl.py").splitlines()[0] + "\n",
        "bitnetd.service.exec": svc,
        "i2sd.py.port": i2,
        "os-release.junior": rd("os-release.junior"),
    }


def preseed_pkgs(text: str) -> list:
    for line in text.splitlines():
        if line.startswith("d-i pkgsel/include string "):
            return line.split("string ", 1)[1].split()
    return []


def preseed_dest(text: str) -> str:
    for line in text.splitlines():
        s = line.lstrip("# ").strip()
        if s.startswith("cp -a ") and s.endswith("/"):
            return s.split()[-1].rstrip("/") or "/"
    return "/"


def land(dest: str, rel: str) -> str:
    return "/" + "/".join(p for p in (dest.strip("/"), rel.strip("/")) if p)


def which(name: str, path: str, present: set) -> str:
    """PATH lookup as string logic against a set of installed absolute paths."""
    for d in path.split(":"):
        if d and land(d, name) in present:
            return land(d, name)
    return ""


def profile_pin(text: str, path: str) -> str:
    """Apply the profile.d case-pin to a PATH string (POSIX sh semantics)."""
    pin = ""
    for line in text.splitlines():
        if line.startswith("JUNIOROS_BIN="):
            pin = line.split("=", 1)[1].strip('"')
    if not pin:
        return path
    return path if (":" + pin + ":") in (":" + path + ":") else pin + ":" + path


def lane(src: dict, base: str, dest: str) -> dict:
    layout = [l for l in src["install-overlay.sh.layout"].splitlines() if l]
    present = {land(dest, rel) for rel in layout}
    prof_at = land(dest, "etc/profile.d/junioros.sh")
    sourced = os.path.dirname(prof_at) == PROFILE_SOURCED
    path = STOCK_PATH[base]
    if sourced:
        path = profile_pin(src["profile.d/junioros.sh"], path)
    unit_at = land(dest, "etc/systemd/system/bitnetd.service")
    unit_seen = os.path.dirname(unit_at) in INIT_SEARCH[base]
    exec_line = src["bitnetd.service.exec"].split("ExecStart=", 1)[1].split("\n", 1)[0]
    return {
        "base": base, "dest": dest, "path": path, "profile_sourced": sourced,
        "juniorctl": which("juniorctl", path, present),
        "unit_seen_by_init": unit_seen,
        "execstart_present": exec_line in present,
    }


def rows_for(src: dict) -> list:
    rows = []

    def add(rid, pair, trit, detail):
        rows.append({"id": rid, "pair": pair, "trit": trit, "detail": detail})

    pkgs = [p for p in src["debian/packages.list"].split() if p]
    pre = preseed_pkgs(src["debian/preseed.fragment"])
    add("pkgs.preseed_subset", "Home>JuniorLLM", 1 if set(pre) <= set(pkgs) else -1,
        "preseed %s within packages.list" % pre)
    gap = sorted(set(pkgs) - set(pre))
    add("pkgs.preseed_gap", "Home>JuniorLLM", 0 if gap else 1,
        "packages.list not in preseed: %s" % (gap or "none"))
    shebang = src["juniorctl.py.shebang"].strip()
    add("juniorctl.shebang", "JuniorLLM>OSai",
        1 if shebang == "#!/usr/bin/env python3" and "python3" in pre else -1,
        "%s with python3 in preseed" % shebang)
    alpine_list = any(k.startswith("alpine/") for k in src)
    add("alpine.lane_files", "JuniorLLM>Flagstaff", 1 if alpine_list else 0,
        "alpine apk list / OpenRC service shipped: %s" % alpine_list)
    prof = src["profile.d/junioros.sh"]
    bashisms = [t for t in ("[[", "function ", "source ", "declare ") if t in prof]
    add("profile.posix_ash", "JuniorLLM>Home", -1 if bashisms else 1,
        "profile.d pin is POSIX (busybox ash + dash): bashisms=%s" % bashisms)

    nd = preseed_dest(src["debian/preseed.fragment"])
    for base in ("debian", "alpine"):
        r = lane(src, base, "/")
        add("%s.dest_root.juniorctl" % base, "Home>OSai", 1 if r["juniorctl"] else -1,
            "DEST=/ juniorctl -> %s" % (r["juniorctl"] or "miss"))
        add("%s.dest_root.execstart" % base, "JuniorLLM>OSai",
            1 if r["execstart_present"] else -1, "DEST=/ junior-bitnetd at ExecStart path")
        add("%s.dest_root.init" % base, "JuniorLLM>Flagstaff",
            1 if r["unit_seen_by_init"] else 0,
            "DEST=/ bitnetd.service seen by %s init: %s"
            % (base, r["unit_seen_by_init"]))
    r = lane(src, "debian", nd)
    add("debian.netinst.juniorctl", "Home>OSai", 1 if r["juniorctl"] else 0,
        "preseed DEST=%s profile.d sourced=%s juniorctl -> %s"
        % (nd, r["profile_sourced"], r["juniorctl"] or "miss on stock PATH"))
    add("debian.netinst.enable", "JuniorLLM>Flagstaff", 1 if r["unit_seen_by_init"] else 0,
        "preseed DEST=%s 'systemctl enable bitnetd.service' finds unit: %s"
        % (nd, r["unit_seen_by_init"]))
    add("debian.netinst.execstart", "JuniorLLM>OSai", 1 if r["execstart_present"] else 0,
        "preseed DEST=%s ExecStart=/usr/local/bin/junior-bitnetd present: %s"
        % (nd, r["execstart_present"]))

    stage = src["debian/stage_overlay.sh"]
    add("stage.i2sd_shipped", "JuniorLLM>FieldCore",
        1 if '"$HERE/i2sd.py"' in stage and "usr/lib/junioros/" in stage else -1,
        "stage_overlay.sh ships i2sd.py under usr/lib/junioros")
    binds = {
        "profile.d": "127.0.0.1:8765" in prof,
        "unit": "JUNIOR_BIND=127.0.0.1:8765" in src["bitnetd.service.exec"],
        "os-release": 'JUNIOR_BITNETD="127.0.0.1:8765"' in src["os-release.junior"],
    }
    add("port.bitnetd_8765", "JuniorLLM>Home", 1 if all(binds.values()) else -1,
        "bitnetd %s agreed: %s" % (BITNETD, binds))
    i2 = src["i2sd.py.port"]
    add("port.i2sd_8767", "JuniorLLM>Home",
        1 if 'HOST = "127.0.0.1"' in i2 and "PORT = 8767" in i2 else -1,
        "i2sd %s declared" % I2SD)
    wild = [k for k, v in src.items() if WILDCARD in v]
    add("bind.no_wildcard", "Home>Flagstaff", -1 if wild else 1,
        "no wildcard bind in sources: %s" % (wild or "clean"))
    flags = (MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH, MODEL_PULL, EXEC)
    add("flags.all_false", "Home>FieldCore",
        1 if not any(flags) and BIND == "127.0.0.1" and RATE == 0.05 else -1,
        "mint/socket/layer_one/fetch/ue5_launch/model_pull/exec false, rate 0.05")
    add("order.live_beta", "OSai>JuniorLLM", 1, " -> ".join(HANDSHAKE_ORDER))
    return rows


def pack(trits) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    out, acc, n = bytearray(), 0, 0
    for t in trits:
        acc = (acc << 2) | code[t]
        n += 1
        if n == 4:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (2 * (4 - n)))
    return bytes(out)


def close(prev: str, packed: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + packed + text.encode()).hexdigest()[:16]


def run(src: dict, prev: str = GENESIS) -> dict:
    rows = rows_for(src)
    text = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    packed = pack([r["trit"] for r in rows])
    return {
        "item": ROADMAP_ITEM, "day": DAY, "ports": PORTS, "rows": rows,
        "pass": sum(r["trit"] == 1 for r in rows),
        "advisory": sum(r["trit"] == 0 for r in rows),
        "fail": sum(r["trit"] == -1 for r in rows),
        "close": close(prev, packed, text),
    }


def selftest() -> int:
    ok = True
    ok &= pack([1, 0, -1, 1]) == bytes([0b01001001])
    ok &= pack([-1]) == bytes([0b10000000])
    ok &= profile_pin(SNAP["profile.d/junioros.sh"], "/usr/bin:/bin") == "/usr/local/bin:/usr/bin:/bin"
    ok &= profile_pin(SNAP["profile.d/junioros.sh"], STOCK_PATH["alpine"]) == STOCK_PATH["alpine"]
    ok &= preseed_dest(SNAP["debian/preseed.fragment"]) == "/opt/junioros"
    ok &= land("/opt/junioros", "usr/local/bin/juniorctl") == "/opt/junioros/usr/local/bin/juniorctl"
    ok &= which("juniorctl", STOCK_PATH["debian"], {"/usr/local/bin/juniorctl"}) == "/usr/local/bin/juniorctl"
    ok &= which("juniorctl", STOCK_PATH["debian"], {"/opt/junioros/usr/local/bin/juniorctl"}) == ""
    a, b = run(SNAP), run(SNAP)
    ok &= a["close"] == b["close"] and len(a["close"]) == 16
    ok &= run(SNAP, prev="f" * 16)["close"] != a["close"]
    bad = dict(SNAP, **{"i2sd.py.port": 'HOST = "%s"\nPORT = 8767\n' % WILDCARD})
    ok &= run(bad)["fail"] >= 2
    ok &= a["fail"] == 0
    print("selftest", "pass" if ok else "fail")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--juniorllm", help="local JuniorLLM tree to read instead of the snapshot")
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    src = load_tree(a.juniorllm) if a.juniorllm else SNAP
    res = run(src, a.prev)
    res["source"] = a.juniorllm or "snapshot@" + SNAPSHOT_TIP
    if a.juniorllm:
        res["drift"] = sorted(
            k for k in SNAP
            if not set(SNAP[k].splitlines()) <= set(src.get(k, "").splitlines()))
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res["rows"]:
            tag = {1: "PASS", 0: "ADV ", -1: "FAIL"}[r["trit"]]
            print("%s %-28s %-20s %s" % (tag, r["id"], r["pair"], r["detail"]))
        print("item: %s" % res["item"])
        print("source: %s  pass %d  advisory %d  fail %d  close %s"
              % (res["source"], res["pass"], res["advisory"], res["fail"], res["close"]))
        if "drift" in res:
            print("drift vs snapshot: %s" % (res["drift"] or "none"))
    return 1 if res["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
