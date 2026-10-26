#!/usr/bin/env python3
"""Not this roadmap: 0.0.0.0 SaaS bind surface port map, read from repo files.

Advances JuniorHome docs/ROADMAP.md named item (section "Not this roadmap"):
    0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user.
Sub-check: "0.0.0.0 SaaS". The existing scripts/ and web3node/ advance_* files
for this item only assert typed constants (SAAS_0000 = False). This port map
reads the actual bind surfaces in this clone, located from this file's path:

  Home      ROADMAP heading + line verbatim; config/*.toml and
            web3node/config/*.toml bind/host keys are 127.0.0.1, bind_all false;
            app/ server binds are loopback.
  OSai      every web3node/ and rails/ server bind literal is loopback; the live
            path (app/serve.py, web3node/trit_tick.py, web3node/os_route.py) and
            its web3node import closure never reaches the legacy SaaS modules
            (community_saas.py, saas_*.py), and those modules bind nothing.
  JuniorLLM docs/TECH_AUDIT.md bind line keeps 8770/8771/8767 on 127.0.0.1,
            "Not 0.0.0.0"; no live-path 0.0.0.0 bind on 8765/8766/8767/8770/8771.
  Flagstaff docs/ZERO_TRUST.md fail list still names "FastAPI host 0.0.0.0:8080"
            and "network_mode: host"; no compose file uses network_mode host;
            no published container port lands on a loopback service port.
  FieldCore every scanned .py parses; legacy src/ 0.0.0.0 defaults and
            container 0.0.0.0 / wide publishes are classified (advisory).

Verdicts: PASS (trit 1), ADVISORY (trit 0, legacy or container surface outside
the live path), FAIL (trit -1, a 0.0.0.0 bind on the live path). Exit 1 on any
FAIL. AST + text reads only: no import of scanned code, no sockets, no network,
no subprocess, no fetch, no model pull, no UE boot. Writes nothing in the repo
(--selftest builds fixtures under a temp dir).

Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false. Rate is a constant; no poll sets it.
Close rule: sha256(prev || pack || text)[:16], balanced trits packed 2-bit
(0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.
Usage: python3 ports/not_this_roadmap_saas_bind_port_map.py [--selftest]
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
import tempfile
import tomllib
import warnings

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
WIDE = "0.0.0.0"
SIDES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ITEM_HEADING = "Not this roadmap"
ITEM = "0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user."
SUBCHECK = "0.0.0.0 SaaS"
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_DIRS = ("app", "web3node", "rails")
LEGACY_DIRS = ("src",)
LIVE_ROOTS = ("app/serve.py", "web3node/trit_tick.py", "web3node/os_route.py")
SAAS_RE = re.compile(r"^(community_saas|saas_\w+)\.py$")
LOOPBACK_PORTS = {8765: "bitnetd", 8766: "deck/app", 8767: "i2sd", 8770: "hook", 8771: "UI"}
HOST_NAMES = {"host", "bind", "HOST", "BIND", "address", "addr"}
CONTAINER_FILES = ("Dockerfile", "docker-compose.yml", "deployment/docker-compose.yml",
                   "deployment/compose.web3node-harness.yml", "deployment/Dockerfile.web3node-harness")
_ENC = {0: 0, 1: 1, -1: 2}
PASS, ADV, FAIL = 1, 0, -1


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


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def _read(root: str, rel: str) -> str:
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _py_files(root: str, top: str):
    base = os.path.join(root, top)
    for d, subs, files in os.walk(base):
        subs[:] = sorted(s for s in subs if not s.startswith(".") and s != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.relpath(os.path.join(d, f), root)


def _const(n):
    return n.value if isinstance(n, ast.Constant) else None


def binds(tree) -> list[tuple[str, object, int]]:
    """(host, port|None, line) for every server-bind literal in a module."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
            pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            vals = {k.arg: _const(d) for k, d in pairs}
            for name in HOST_NAMES:
                if isinstance(vals.get(name), str):
                    out.append((vals[name], vals.get("port"), node.lineno))
        elif isinstance(node, ast.Call):
            kw = {k.arg: _const(k.value) for k in node.keywords if k.arg}
            for name in HOST_NAMES:
                if isinstance(kw.get(name), str):
                    out.append((kw[name], kw.get("port"), node.lineno))
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname.endswith("Server") and node.args and isinstance(node.args[0], ast.Tuple):
                el = node.args[0].elts
                if len(el) == 2 and isinstance(_const(el[0]), str):
                    out.append((_const(el[0]), _const(el[1]), node.lineno))
        elif isinstance(node, ast.Assign) and isinstance(_const(node.value), str):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in HOST_NAMES:
                    out.append((node.value.value, None, node.lineno))
    return out


def _split(host: str, port, line: int):
    if port is None and host.count(":") == 1 and host.rsplit(":", 1)[1].isdigit():
        host, port = host.rsplit(":", 1)[0], int(host.rsplit(":", 1)[1])
    return host, port, line


def imports(tree) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module.split(".")[0])
    return out


def _toml_hosts(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _toml_hosts(v, f"{path}.{k}" if path else k)
    elif path.rsplit(".", 1)[-1] in ("bind", "host", "bind_all"):
        yield path, obj


def scan(root: str) -> dict:
    trees, bad = {}, []
    for top in LIVE_DIRS + LEGACY_DIRS:
        for rel in _py_files(root, top):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trees[rel] = ast.parse(_read(root, rel), filename=rel)
            except (SyntaxError, ValueError):
                bad.append(rel)
    found = {rel: [_split(h, p, ln) for h, p, ln in binds(t)] for rel, t in trees.items()}
    w3 = {os.path.basename(r)[:-3]: r for r in trees if r.startswith("web3node/") and r.count("/") == 1}
    saas = {m for m, r in w3.items() if SAAS_RE.match(os.path.basename(r))}
    seen, todo = set(), [r for r in LIVE_ROOTS if r in trees]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        todo += [w3[m] for m in imports(trees[rel]) if m in w3]
    importers = sorted(r for r in w3.values() if imports(trees[r]) & saas and os.path.basename(r)[:-3] not in saas)
    tomls = {}
    for rel in ("config", "web3node/config"):
        d = os.path.join(root, rel)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".toml"):
                try:
                    tomls[f"{rel}/{f}"] = list(_toml_hosts(tomllib.loads(_read(root, f"{rel}/{f}"))))
                except tomllib.TOMLDecodeError:
                    bad.append(f"{rel}/{f}")
    cont = {f: _read(root, f) for f in CONTAINER_FILES if os.path.exists(os.path.join(root, f))}
    return {"trees": trees, "bad": bad, "binds": found, "saas": saas, "w3": w3,
            "closure": seen, "importers": importers, "tomls": tomls, "cont": cont,
            "roadmap": _read(root, "docs/ROADMAP.md"), "tech": _read(root, "docs/TECH_AUDIT.md"),
            "zt": _read(root, "docs/ZERO_TRUST.md")}


def checks(s: dict) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    add = lambda side, name, v: rows.append((side, name, v))  # noqa: E731
    ok = lambda b: PASS if b else FAIL  # noqa: E731
    rm = s["roadmap"]
    sec = rm.split(f"## {ITEM_HEADING}", 1)
    add("Home", f'ROADMAP "## {ITEM_HEADING}" names "{ITEM}"', ok(len(sec) == 2 and ITEM in sec[1]))
    add("Home", f'ROADMAP sub-check "{SUBCHECK}" present', ok(SUBCHECK in rm))
    for rel, kv in s["tomls"].items():
        for key, val in kv:
            good = (val is False) if key.endswith("bind_all") else (val == BIND)
            add("Home", f"{rel} {key} = {val!r}", ok(good))
    if not s["tomls"]:
        add("Home", "config/*.toml bind keys found", FAIL)
    live = {r: b for r, b in s["binds"].items() if r.split("/")[0] in LIVE_DIRS}
    consts: dict[str, list[str]] = {"Home": [], "OSai": []}
    for rel, bl in sorted(live.items()):
        side = "Home" if rel.startswith("app/") else "OSai"
        for host, port, line in bl:
            if port is None:
                consts[side].append(f"{rel}:{line}" if host == WIDE else "")
            else:
                add(side, f"{rel}:{line} server bind {host}:{port}", ok(host != WIDE))
    for side, lst in consts.items():
        if lst:
            wide = [x for x in lst if x]
            add(side, f"{len(lst)} portless host/BIND literals on live dirs all {BIND}"
                + (f"; wide at {wide}" if wide else ""), ok(not wide))
    if not any(live.values()):
        add("OSai", "live-path server bind literals found", ADV)
    hit = sorted(set(s["closure"]) & {s["w3"][m] for m in s["saas"]})
    add("OSai", f"live path closure ({len(s['closure'])} modules) excludes legacy SaaS {sorted(s['saas'])}",
        ok(not hit and bool(s["closure"])))
    for m in sorted(s["saas"]):
        rel = s["w3"][m]
        add("OSai", f"legacy SaaS {rel} binds nothing", ok(not s["binds"].get(rel)))
    for rel in s["importers"]:
        add("OSai", f"{rel} imports legacy SaaS (off live path, no bind)", ADV)
    tech = s["tech"]
    bl = next((ln for ln in tech.splitlines() if ln.startswith("- Bind:")), "")
    add("JuniorLLM", "TECH_AUDIT bind line: 127.0.0.1 8770/8771/8767, Not 0.0.0.0",
        ok(all(p in bl for p in ("127.0.0.1:8770", "8771", "8767", "Not 0.0.0.0"))))
    for port, svc in sorted(LOOPBACK_PORTS.items()):
        wide_live = [f"{r}:{ln}" for r, b in live.items() for h, p, ln in b if h == WIDE and p == port]
        add("JuniorLLM", f"{svc} {port}: no live 0.0.0.0 bind", ok(not wide_live))
    legacy = {r: b for r, b in s["binds"].items() if r.split("/")[0] in LEGACY_DIRS}
    for rel, b in sorted(legacy.items()):
        for host, port, line in b:
            if host == WIDE:
                extra = f" (shares {LOOPBACK_PORTS[port]} port {port})" if port in LOOPBACK_PORTS else ""
                add("FieldCore", f"legacy {rel}:{line} default {host}:{port}{extra}", ADV)
    add("FieldCore", f"{len(s['trees'])} .py + {len(s['tomls'])} .toml parsed, unparsed {s['bad'] or 0}",
        PASS if not s["bad"] else ADV)
    zt = s["zt"]
    add("Flagstaff", 'ZERO_TRUST fail list names "FastAPI host 0.0.0.0:8080"', ok("FastAPI host 0.0.0.0:8080" in zt))
    add("Flagstaff", 'ZERO_TRUST fail list names "network_mode: host"', ok("network_mode: host" in zt))
    for f, txt in sorted(s["cont"].items()):
        if f.endswith(".yml"):
            add("Flagstaff", f"{f} no network_mode host", ok(not re.search(r"network_mode:\s*['\"]?host", txt)))
        for m in re.finditer(r"--host[\"',\s]+([\d.]+)[\"',\s]+--port[\"',\s]+(\d+)", txt):
            if m.group(1) == WIDE:
                add("Flagstaff", f"{f} container listens {WIDE}:{m.group(2)} (container-internal, legacy)", ADV)
        for m in re.finditer(r"^\s*-\s*['\"]?(?:([\d.]+):)?(\d+):(\d+)['\"]?\s*$", txt, re.M):
            ip, hp = m.group(1), int(m.group(2))
            add("Flagstaff", f"{f} publish {m.group(0).strip()} not on a loopback service port",
                ok(hp not in LOOPBACK_PORTS))
            if ip != BIND:
                add("Flagstaff", f"{f} publishes {hp} on all interfaces (no {BIND}: prefix)", ADV)
    add("Flagstaff", "held: opt_in anticon pool_held, rate 0.05",
        ok(OPT_IN and ANTICON and POOL_HELD and RATE == 0.05 and BIND == "127.0.0.1"))
    add("Flagstaff", "mint socket layer_one fetch ue5_launch model_pull false",
        ok(not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH or MODEL_PULL)))
    return rows


def report(root: str, quiet: bool = False) -> tuple[int, int, int, str]:
    rows = checks(scan(root))
    pk = pack([v for *_, v in rows])
    tip = GENESIS
    for side in SIDES:
        mine = [r for r in rows if r[0] == side]
        text = f"{ITEM_HEADING}|{SUBCHECK}|{side}|" + ";".join(f"{n}={v}" for _, n, v in mine)
        tip = close(tip, pk, text)
        if not quiet:
            for _, name, v in mine:
                print(f"{ {PASS: 'PASS', ADV: 'ADVISORY', FAIL: 'FAIL'}[v] } [{side}] {name}")
    p = sum(v == PASS for *_, v in rows)
    a = sum(v == ADV for *_, v in rows)
    f = sum(v == FAIL for *_, v in rows)
    return p, a, f, tip


def run() -> int:
    p, a, f, tip = report(ROOT)
    print(f'{ITEM_HEADING}: "{ITEM}" [{SUBCHECK}] pass {p} advisory {a} fail {f} '
          f"close {tip} rate {RATE} bind {BIND}")
    return 1 if f else 0


def _fixture(d: str, **over) -> None:
    files = {
        "docs/ROADMAP.md": f"# r\n## Order\n1. x\n\n## {ITEM_HEADING}\n{ITEM}\n",
        "docs/TECH_AUDIT.md": "- Bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd. Not 0.0.0.0.\n",
        "docs/ZERO_TRUST.md": "## Fail\n- FastAPI host 0.0.0.0:8080\n- docker-compose network_mode: host\n",
        "config/deck_port.toml": 'bind = "127.0.0.1"\nbind_all = false\n',
        "app/serve.py": "import trit_tick\nfrom http.server import ThreadingHTTPServer\n"
                        "def go():\n    ThreadingHTTPServer(('127.0.0.1', 8766), None)\n",
        "web3node/trit_tick.py": "import json\n",
        "web3node/os_route.py": "import trit_tick\n",
        "web3node/community_saas.py": "def credit():\n    return 1\n",
        "web3node/missing_live.py": "from community_saas import credit\n",
        "src/juniorhome/ws.py": "class W:\n    def __init__(self, host='0.0.0.0', port=8765):\n        pass\n",
        "Dockerfile": 'CMD ["uvicorn", "a:app", "--host", "0.0.0.0", "--port", "8000"]\n',
        "docker-compose.yml": 'services:\n  a:\n    ports:\n      - "8000:8000"\n',
    }
    files.update(over)
    for rel, body in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)


def selftest() -> int:
    assert pack([1, -1, 0, 1]) == bytes([0b01100001])
    assert pack([1, 1, 1, 1, -1]) == bytes([0b01010101, 0b10000000])
    c1 = close(GENESIS, pack([1]), "a")
    det = c1 == close(GENESIS, pack([1]), "a") and c1 != close(GENESIS, pack([-1]), "a")
    cases = {
        "clean": ({}, lambda p, a, f: f == 0 and a == 4 and p > 0),
        "live_wide_bind": ({"web3node/status_api.py": "from http.server import HTTPServer\n"
                            "HTTPServer(('0.0.0.0', 8765), None)\n"}, lambda p, a, f: f == 2),
        "toml_bind_all": ({"config/deck_port.toml": 'bind = "127.0.0.1"\nbind_all = true\n'},
                          lambda p, a, f: f == 1),
        "saas_on_live_path": ({"web3node/os_route.py": "import trit_tick\nimport community_saas\n"},
                              lambda p, a, f: f == 1),
        "saas_binds": ({"web3node/saas_loop.py": "def serve(host='0.0.0.0', port=9000):\n    pass\n"},
                       lambda p, a, f: f == 2),
        "network_mode_host": ({"docker-compose.yml": "services:\n  a:\n    network_mode: host\n"},
                              lambda p, a, f: f == 1),
        "publish_on_loopback_port": ({"docker-compose.yml": 'services:\n  a:\n    ports:\n      - "8767:8767"\n'},
                                     lambda p, a, f: f == 1),
        "roadmap_item_dropped": ({"docs/ROADMAP.md": "# r\n## Order\n1. x\n"}, lambda p, a, f: f == 2),
        "loopback_publish_ok": ({"docker-compose.yml": 'services:\n  a:\n    ports:\n      - "127.0.0.1:8000:8000"\n'},
                                lambda p, a, f: f == 0 and a == 3),
    }
    res = {}
    for name, (over, want) in cases.items():
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **over)
            p, a, f, tip = report(d, quiet=True)
            res[name] = bool(want(p, a, f)) and len(tip) == 16
            print(f"selftest {name}: pass {p} advisory {a} fail {f} close {tip} -> {'ok' if res[name] else 'BAD'}")
    good = det and all(res.values())
    print(f"selftest {'pass' if good else 'fail'}: {sum(res.values())}/{len(res)} cases, close_det={det}")
    return 0 if good and run() == 0 else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else run())
