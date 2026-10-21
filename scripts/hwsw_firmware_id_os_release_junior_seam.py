#!/usr/bin/env python3
"""Hardware-to-software seam emulator: firmware identity (devicetree model / DMI) -> os-release.junior + bitnetd.service.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md, "JuniorOS overlay — Debian/Alpine +
  os-release.junior + hardened bitnetd.service + juniorctl on PATH", on the
  hardware class rows from JuniorHome docs/HW_CLASS.md: SBC (arm64, <=12 W ->
  JuniorOS T4) and SFF host (x86_64 or arm64 -> JuniorHome T0). Asahi rows
  touch ROADMAP item 4 as probe only (mlx_admit is always False). The seam
  answers one question: given what the firmware says the box is, does the
  overlay's os-release.junior name the right class and tier, does
  bitnetd.service stay on 127.0.0.1:8765 with its hardening keys, and is
  juniorctl on PATH? Not ROADMAP T13 (overnight owns it).

What it emulates
  In-memory fixtures shaped like /proc/device-tree/model, the
  /sys/class/dmi/id/{sys_vendor,product_name,chassis_type} files, a watt
  class, an os-release.junior text, a bitnetd.service unit text, a PATH
  string and a set of executable paths.
  1. Class: arm64 with a devicetree model and <=12 W -> sbc (JuniorOS T4).
     x86_64 with a DMI chassis_type in {3,4,6,7,35,36} -> sff (JuniorHome T0).
     arm64 with an Apple devicetree model and >12 W -> sff with asahi=True
     (MLX probe only). Anything else -> unknown, and the row rejects.
  2. os-release.junior: KEY=value lines (quotes stripped). Needs ID=junioros,
     ID_LIKE debian or alpine, JUNIOR_CLASS and JUNIOR_TIER equal to the
     derived class and tier, JUNIOR_BITNETD=127.0.0.1:8765. Per HW_CLASS.md
     ("name the class, not a vendor box") no value may contain the firmware
     vendor or product string.
  3. bitnetd.service: [Service] ExecStart must carry --host 127.0.0.1 and
     --port 8765; NoNewPrivileges=yes, ProtectSystem=strict, PrivateDevices=yes,
     IPAddressDeny=any, IPAddressAllow=localhost; RestrictAddressFamilies may
     not list AF_PACKET or AF_NETLINK. Any 0.0.0.0 rejects.
  4. juniorctl: one PATH entry joined with "juniorctl" must be in the
     executable set.
  Each check is a trit (+1 pass, 0 not applicable, -1 fail), packed four per
  byte low bits first (codes 0/1/2 = -1/0/+1). One row per host fixture is
  chained with close rule sha256(prev || pack || text)[:16] at rate 0.05,
  where pack is the packed check trits plus a class byte.

What it does not touch
  No /proc, /sys, /etc, systemd, dmidecode, devicetree overlay or real unit
  file is read or written; no systemctl call. No socket is opened: binds are
  strings and are never bound. It does not import or re-run bitnetd, i2sd,
  juniorctl, scripts/advance_*, ports/*, OSai bench notes, or any other team
  file, and it does not cover the earlier 06:00 seams (spool, USB sound/HID,
  SBC T4, SFF T0, USB-C, tun, CPU ISA). No model pull, no GGUF. Rows go to a
  tempfile removed after the run. Standard library only.
  Flags: mint, socket, layer_one, fetch, ue5_launch False.

How to run
  python3 scripts/hwsw_firmware_id_os_release_junior_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_firmware_id_os_release_junior_seam.py demo     # emulated host rows
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
BITNETD_HOST = "127.0.0.1"
BITNETD_PORT = "8765"
BITNETD_BIND = BITNETD_HOST + ":" + BITNETD_PORT
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16

SFF_CHASSIS = {3, 4, 6, 7, 35, 36}
TIER = {"sbc": "JuniorOS T4", "sff": "JuniorHome T0"}
CLASS_BYTE = {"sbc": 1, "sff": 2, "unknown": 0}
HARDEN = {
    "NoNewPrivileges": "yes",
    "ProtectSystem": "strict",
    "PrivateDevices": "yes",
    "IPAddressDeny": "any",
    "IPAddressAllow": "localhost",
}
BANNED_AF = ("AF_PACKET", "AF_NETLINK")
CHECKS = ("class", "os_release", "vendor_free", "bitnetd_bind", "bitnetd_harden", "juniorctl", "asahi_probe")


def classify(host):
    arch = host.get("arch")
    model = (host.get("dt_model") or "").strip("\x00 ").strip()
    dmi = host.get("dmi") or {}
    watts = host.get("watts", 0)
    if arch == "aarch64" and model and model.lower().startswith("apple") and watts > 12:
        return "sff", True
    if arch == "aarch64" and model and watts <= 12:
        return "sbc", False
    if arch == "x86_64":
        try:
            chassis = int(str(dmi.get("chassis_type", "")).strip())
        except ValueError:
            chassis = -1
        if chassis in SFF_CHASSIS:
            return "sff", False
    return "unknown", False


def parse_os_release(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def parse_unit(text):
    section, out = None, {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out.setdefault(section, {})[k.strip()] = v.strip()
    return out


def vendor_tokens(host):
    toks = []
    dmi = host.get("dmi") or {}
    for key in ("sys_vendor", "product_name"):
        v = str(dmi.get(key, "")).strip()
        if len(v) >= 3:
            toks.append(v.lower())
    model = (host.get("dt_model") or "").strip("\x00 ").strip()
    if model:
        toks.append(model.split()[0].lower())
    return toks


def check_host(host):
    cls, asahi = classify(host)
    rel = parse_os_release(host.get("os_release_junior", ""))
    unit = parse_unit(host.get("bitnetd_unit", "")).get("Service", {})
    t = {}
    t["class"] = 1 if cls != "unknown" else -1
    ok = (
        rel.get("ID") == "junioros"
        and rel.get("ID_LIKE") in ("debian", "alpine")
        and cls != "unknown"
        and rel.get("JUNIOR_CLASS") == cls
        and rel.get("JUNIOR_TIER") == TIER.get(cls)
        and rel.get("JUNIOR_BITNETD") == BITNETD_BIND
    )
    t["os_release"] = 1 if ok else -1
    toks = vendor_tokens(host)
    leak = [v for v in rel.values() for tok in toks if tok and tok in v.lower()]
    t["vendor_free"] = -1 if leak else 1
    argv = unit.get("ExecStart", "").split()
    def arg(name):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None
    blob = host.get("bitnetd_unit", "") + host.get("os_release_junior", "")
    bind_ok = arg("--host") == BITNETD_HOST and arg("--port") == BITNETD_PORT and "0.0.0.0" not in blob
    t["bitnetd_bind"] = 1 if bind_ok else -1
    af = unit.get("RestrictAddressFamilies", "")
    harden_ok = all(unit.get(k) == v for k, v in HARDEN.items()) and not any(b in af for b in BANNED_AF)
    t["bitnetd_harden"] = 1 if harden_ok else -1
    exe = set(host.get("executables", ()))
    on_path = any(os.path.join(p, "juniorctl") in exe for p in host.get("path", "").split(":") if p)
    t["juniorctl"] = 1 if on_path else -1
    if asahi:
        t["asahi_probe"] = 1 if rel.get("JUNIOR_MLX", "probe") == "probe" else -1
    else:
        t["asahi_probe"] = 0
    trits = [t[c] for c in CHECKS]
    admit = all(x != -1 for x in trits)
    return {
        "host": host["name"],
        "class": cls,
        "tier": TIER.get(cls),
        "asahi": asahi,
        "mlx_admit": False,
        "checks": t,
        "admit": admit,
        "rate": RATE,
        "flags": dict(FLAGS),
    }, trits


def pack_trits(trits, cls):
    codes = [x + 1 for x in trits]
    while len(codes) % 4:
        codes.append(1)
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            b |= (c & 3) << (2 * j)
        out.append(b)
    out.append(CLASS_BYTE[cls])
    return bytes(out)


def close(prev, pack, text):
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def run(hosts):
    prev, rows = GENESIS, []
    for h in hosts:
        row, trits = check_host(h)
        text = json.dumps(row, sort_keys=True)
        row["close"] = close(prev, pack_trits(trits, row["class"]), text)
        row["prev"] = prev
        prev = row["close"]
        rows.append(row)
    return rows


def _rel(cls, extra=""):
    return (
        'ID=junioros\nID_LIKE=debian\nNAME="JuniorOS overlay"\n'
        "JUNIOR_CLASS=%s\nJUNIOR_TIER=\"%s\"\nJUNIOR_BITNETD=%s\n%s" % (cls, TIER[cls], BITNETD_BIND, extra)
    )


def _unit(host=BITNETD_HOST, extra=""):
    return (
        "[Unit]\nDescription=bitnetd loopback\n\n[Service]\n"
        "ExecStart=/usr/local/bin/bitnetd --host %s --port %s\n"
        "NoNewPrivileges=yes\nProtectSystem=strict\nPrivateDevices=yes\n"
        "IPAddressDeny=any\nIPAddressAllow=localhost\n"
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6\n%s\n"
        "[Install]\nWantedBy=multi-user.target\n" % (host, BITNETD_PORT, extra)
    )


PATH = "/usr/local/sbin:/usr/local/bin:/usr/bin:/bin"
EXE = ("/usr/local/bin/juniorctl", "/usr/local/bin/bitnetd")


def fixtures():
    sbc = {"name": "sbc_ok", "arch": "aarch64", "dt_model": "Generic RK3588 board\x00", "dmi": {},
           "watts": 10, "os_release_junior": _rel("sbc"), "bitnetd_unit": _unit(), "path": PATH, "executables": EXE}
    sff = {"name": "sff_x86_ok", "arch": "x86_64", "dt_model": None,
           "dmi": {"sys_vendor": "ExampleCorp", "product_name": "MiniBox 5", "chassis_type": "35"},
           "watts": 35, "os_release_junior": _rel("sff"), "bitnetd_unit": _unit(), "path": PATH, "executables": EXE}
    asahi = {"name": "sff_asahi_probe", "arch": "aarch64", "dt_model": "Apple Mac mini (M2)\x00", "dmi": {},
             "watts": 40, "os_release_junior": _rel("sff", "JUNIOR_MLX=probe\n"), "bitnetd_unit": _unit(),
             "path": PATH, "executables": EXE}
    leak = dict(sff, name="vendor_leak", os_release_junior=_rel("sff", 'PRETTY_NAME="ExampleCorp MiniBox 5"\n'))
    wide = dict(sbc, name="bind_wide", bitnetd_unit=_unit(host="0.0.0.0"))
    wrong = dict(sff, name="tier_mismatch", os_release_junior=_rel("sbc"))
    noctl = dict(sbc, name="juniorctl_missing", path="/usr/bin:/bin")
    soft = dict(sbc, name="harden_af_packet", bitnetd_unit=_unit().replace("AF_INET6", "AF_INET6 AF_PACKET"))
    return [sbc, sff, asahi, leak, wide, wrong, noctl, soft]


def self_test():
    hosts = fixtures()
    rows = run(hosts)
    want = {"sbc_ok": True, "sff_x86_ok": True, "sff_asahi_probe": True, "vendor_leak": False,
            "bind_wide": False, "tier_mismatch": False, "juniorctl_missing": False, "harden_af_packet": False}
    results = []
    for r in rows:
        results.append({"host": r["host"], "admit": r["admit"], "pass": r["admit"] == want[r["host"]]})
    again = run(hosts)
    results.append({"host": "chain_deterministic", "pass": [r["close"] for r in again] == [r["close"] for r in rows]})
    results.append({"host": "mlx_never_admitted", "pass": all(r["mlx_admit"] is False for r in rows)})
    results.append({"host": "flags_false", "pass": all(not v for r in rows for v in r["flags"].values())})
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    finally:
        os.remove(path)
    ok = all(x["pass"] for x in results)
    print(json.dumps({"seam": "firmware_id_os_release_junior", "rate": RATE, "bitnetd": BITNETD_BIND,
                      "rows": results, "tip_close": rows[-1]["close"], "ok": ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print(json.dumps(run(fixtures()), indent=2, sort_keys=True))
        sys.exit(0)
    sys.exit(self_test())
