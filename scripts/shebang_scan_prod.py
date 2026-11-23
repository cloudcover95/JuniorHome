#!/usr/bin/env python3
"""shebang_scan_prod.py - read-only scan of scripts/*_prod.py for shebang and
executable-bit consistency. Stdlib only. No network, no writes.
Optional --serve binds 127.0.0.1 only and returns the JSON report."""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

def scan(root):
    rows = []
    d = os.path.join(root, "scripts")
    names = sorted(n for n in os.listdir(d) if n.endswith("_prod.py")) if os.path.isdir(d) else []
    for n in names:
        p = os.path.join(d, n)
        with open(p, "rb") as f:
            first = f.readline(200).decode("utf-8", "replace").strip()
        rows.append({"file": n, "shebang": first if first.startswith("#!") else "",
                     "exec": os.access(p, os.X_OK)})
    return {"count": len(rows),
            "no_shebang": [r["file"] for r in rows if not r["shebang"]],
            "exec_without_shebang": [r["file"] for r in rows if r["exec"] and not r["shebang"]],
            "shebang_not_exec": [r["file"] for r in rows if r["shebang"] and not r["exec"]]}

def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only shebang/exec-bit scan of scripts/*_prod.py")
    ap.add_argument("--root", default=".", help="repo root (default .)")
    ap.add_argument("--serve", type=int, default=0, help="serve JSON on 127.0.0.1:PORT")
    a = ap.parse_args(argv)
    if a.serve:
        root = a.root
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                b = json.dumps(scan(root)).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        HTTPServer(("127.0.0.1", a.serve), H).serve_forever()
        return 0
    print(json.dumps(scan(a.root), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
