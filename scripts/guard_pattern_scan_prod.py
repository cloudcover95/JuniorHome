#!/usr/bin/env python3
"""guard_pattern_scan_prod.py - read-only scan of scripts/*_prod.py for LLC guard violations.

Checks (guards 3 and 8): no 0.0.0.0 bind, no model pull / GGUF download
markers, and no .github path references. Reports JSON. Writes nothing.
Stdlib only. Optional --serve exposes the report on 127.0.0.1 only.
"""
import argparse
import ast
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PATTERNS = {
    "bind_all": re.compile(r"0\.0\.0\.0"),
    "gguf_pull": re.compile(r"\.gguf|huggingface\.co|ollama\s+pull", re.I),
    "github_dir": re.compile(r"\.github/"),
}


def scan(root):
    out = []
    for p in sorted(Path(root).glob("*_prod.py")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            ast.parse(text)
        except SyntaxError as e:
            out.append({"file": p.name, "rule": "syntax", "line": e.lineno or 0})
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for rule, rx in PATTERNS.items():
                if rx.search(line):
                    out.append({"file": p.name, "rule": rule, "line": n})
    return out


def build_report(root):
    files = len(list(Path(root).glob("*_prod.py")))
    hits = scan(root)
    return {"scanned": files, "hits": len(hits), "findings": hits, "ok": not hits}


def serve(root, port):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(build_report(root)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    HTTPServer(("127.0.0.1", port), H).serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only guard scan of scripts/*_prod.py.")
    ap.add_argument("--root", default="scripts", help="folder holding *_prod.py")
    ap.add_argument("--serve", type=int, metavar="PORT", help="serve report on 127.0.0.1")
    a = ap.parse_args(argv)
    if a.serve:
        serve(a.root, a.serve)
        return 0
    rep = build_report(a.root)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
