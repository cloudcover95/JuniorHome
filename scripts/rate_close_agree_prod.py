#!/usr/bin/env python3
"""rate_close_agree_prod.py - JuniorHome guard 5/7 agreement check.

Read-only. Stdlib only. JuniorCloud LLC. cloudcover95.

Walks a tree of Python files and reports whether the engines still agree on
the shared trit-pack constants from docs/LLC_GUARDS.md:

  * rate 0.05
  * close rule sha256(prev || pack || text)[:16]
  * guard flags mint/socket/layer_one/fetch/ue5_launch all false

It never edits, never fetches, never opens a socket. The optional --serve
mode is a tiny JSON report on 127.0.0.1 only.

Usage:
  python3 rate_close_agree_prod.py [--root DIR] [--json]
  python3 rate_close_agree_prod.py --close PREV PACK TEXT
  python3 rate_close_agree_prod.py --selftest
"""
import argparse
import ast
import hashlib
import json
import os
import sys

RATE = 0.05
CLOSE_LEN = 16
FLAGS = ("mint", "socket", "layer_one", "fetch", "ue5_launch")
HOST = "127.0.0.1"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def close_rule(prev, pack, text):
    """Reference close: sha256(prev || pack || text)[:16] as hex."""
    h = hashlib.sha256()
    for part in (prev, pack, text):
        h.update(part.encode("utf-8") if isinstance(part, str) else bytes(part))
    return h.hexdigest()[:CLOSE_LEN]


def _const(node):
    if isinstance(node, ast.Constant):
        return node.value
    return None


def scan_source(src, name="<src>"):
    """Return findings for one source text."""
    out = {"file": name, "rates": [], "flags": {}, "close_slices": [], "error": None}
    try:
        tree = ast.parse(src, filename=name)
    except SyntaxError as exc:
        out["error"] = "syntax: %s" % exc.msg
        return out
    for node in ast.walk(tree):
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                key = _const(k) if k is not None else None
                if isinstance(key, str):
                    _record(out, key, _const(v))
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                _record(out, t.id, _const(value))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            sl = node.slice
            base = node.value
            is_hex = (isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute)
                      and base.func.attr == "hexdigest")
            if is_hex and sl.lower is None and isinstance(_const(sl.upper), int):
                out["close_slices"].append(_const(sl.upper))
    return out


def _record(out, key, val):
    k = key.lower()
    if k in ("rate", "trit_rate", "pack_rate", "rate_default") and isinstance(val, (int, float)) \
            and not isinstance(val, bool):
        out["rates"].append(float(val))
    if k in FLAGS and isinstance(val, bool):
        out["flags"][k] = val


def iter_py(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def scan_tree(root):
    files = []
    for path in iter_py(root):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError as exc:
            files.append({"file": path, "error": str(exc), "rates": [], "flags": {}, "close_slices": []})
            continue
        f = scan_source(src, os.path.relpath(path, root))
        if f["rates"] or f["flags"] or f["close_slices"] or f["error"]:
            files.append(f)
    return summarize(files)


def summarize(files):
    bad = []
    for f in files:
        for r in f["rates"]:
            if abs(r - RATE) > 1e-12:
                bad.append("%s: rate %r != %r" % (f["file"], r, RATE))
        for k, v in sorted(f["flags"].items()):
            if v is not False:
                bad.append("%s: %s is %r, want False" % (f["file"], k, v))
        for n in f["close_slices"]:
            if n != CLOSE_LEN:
                bad.append("%s: hexdigest()[:%d] != [:%d] (check if close rule)" % (f["file"], n, CLOSE_LEN))
    return {
        "rate": RATE,
        "close_len": CLOSE_LEN,
        "files_with_findings": len(files),
        "disagreements": bad,
        "agree": not bad,
        "files": files,
    }


def selftest():
    a = close_rule("p", "k", "t")
    assert a == hashlib.sha256(b"pkt").hexdigest()[:16], a
    assert len(a) == 16
    good = "rate = 0.05\nmint = False\nx = hashlib.sha256(b'').hexdigest()[:16]\n"
    r = summarize([scan_source(good, "good.py")])
    assert r["agree"], r
    badsrc = "RATE = 0.07\nsocket = True\nh = hashlib.sha256(b'').hexdigest()[:12]\ncfg = {'fetch': True}\n"
    r = summarize([scan_source(badsrc, "bad.py")])
    assert not r["agree"] and len(r["disagreements"]) == 4, r
    assert scan_source("def (:", "x.py")["error"]
    return True


def serve(root, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(scan_tree(root), indent=1).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer((HOST, port), H)
    print("serving on http://%s:%d/" % (HOST, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check that JuniorHome engines agree on rate 0.05, "
                                 "the sha256(prev||pack||text)[:16] close rule and false guard flags. Read-only.")
    ap.add_argument("--root", default=".", help="tree to scan (default: .)")
    ap.add_argument("--json", action="store_true", help="print full JSON report")
    ap.add_argument("--close", nargs=3, metavar=("PREV", "PACK", "TEXT"), help="print reference close hash")
    ap.add_argument("--selftest", action="store_true", help="run built-in checks")
    ap.add_argument("--serve", type=int, metavar="PORT", help="serve JSON report on 127.0.0.1:PORT")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        print("selftest ok")
        return 0
    if a.close:
        print(close_rule(*a.close))
        return 0
    if a.serve:
        serve(a.root, a.serve)
        return 0
    rep = scan_tree(a.root)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print("rate %.2f close [:%d] files %d agree %s" % (RATE, CLOSE_LEN, rep["files_with_findings"], rep["agree"]))
        for line in rep["disagreements"]:
            print("  " + line)
    return 0 if rep["agree"] else 1


if __name__ == "__main__":
    sys.exit(main())
