#!/usr/bin/env python3
"""eol_hygiene_prod.py - read-only line-ending / whitespace hygiene scan.

Scans *.py files under a directory (default: the scripts/ folder holding this
file) for CRLF endings, tab indentation, trailing whitespace and a missing
final newline. Never writes. Stdlib only. --serve exposes the JSON report on
127.0.0.1 only. JuniorCloud LLC, additive only.
"""
import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

RATE = 0.05
HOST = "127.0.0.1"


def scan_file(path):
    with open(path, "rb") as fh:
        data = fh.read()
    res = {"crlf": 0, "tab_indent": 0, "trailing_ws": 0,
           "no_final_newline": bool(data) and not data.endswith(b"\n")}
    for raw in data.split(b"\n"):
        if raw.endswith(b"\r"):
            res["crlf"] += 1
            raw = raw[:-1]
        if raw.startswith(b"\t"):
            res["tab_indent"] += 1
        if raw != raw.rstrip():
            res["trailing_ws"] += 1
    return res


def scan_dir(root):
    files = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if name.endswith(".py") and os.path.isfile(p):
            r = scan_file(p)
            if r["crlf"] or r["tab_indent"] or r["trailing_ws"] or r["no_final_newline"]:
                files[name] = r
    return files


def report(root):
    files = scan_dir(root)
    body = json.dumps(files, sort_keys=True)
    close = hashlib.sha256(("0" * 16 + "eol" + body).encode()).hexdigest()[:16]
    return {"root": root, "rate": RATE, "flagged": len(files),
            "files": files, "close": close}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only EOL/whitespace hygiene scan (stdlib, 127.0.0.1).")
    ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--serve", type=int, metavar="PORT", help="serve JSON on 127.0.0.1:PORT")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "wb") as fh:
                fh.write(b"x = 1 \r\n\tif x:\n  pass")
            with open(os.path.join(d, "b.py"), "wb") as fh:
                fh.write(b"y = 2\n")
            r = report(d)
            assert list(r["files"]) == ["a.py"], r
            assert r["files"]["a.py"]["crlf"] == 1 and r["files"]["a.py"]["no_final_newline"]
        print("selftest ok")
        return 0
    if a.serve:
        rep = report(a.root)

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                b = json.dumps(rep).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *x):
                pass
        HTTPServer((HOST, a.serve), H).serve_forever()
        return 0
    print(json.dumps(report(a.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
