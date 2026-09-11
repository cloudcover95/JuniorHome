#!/usr/bin/env python3
"""Loopback UI. 127.0.0.1:8772"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOME = Path(__file__).resolve().parents[1]
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "user_in.py").is_file():
        sys.path.insert(0, str(p))
        break

from ports.flagstaff_balance import guess
from ports.user_in import ingest

VAULT = HOME / "vault"
PAGE = (HOME / "ui" / "index.html").read_text(encoding="utf-8")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        return

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path in {"/", "/index.html"}:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if u.path == "/guess":
            q = (parse_qs(u.query).get("q") or [""])[0]
            self._send(200, json.dumps({"area": guess(q), "q": q}).encode())
            return
        self._send(404, b"{}")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode() if n else "{}"
        try:
            body = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._send(400, b"{}")
            return
        if urlparse(self.path).path == "/inject":
            out = ingest(VAULT, str(body.get("text") or ""))
            self._send(200, json.dumps(out).encode())
            return
        self._send(404, b"{}")


def serve(host: str = "127.0.0.1", port: int = 8772) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("loopback only")
    return ThreadingHTTPServer((host, port), H)


def main() -> int:
    print("ui 127.0.0.1:8772")
    serve().serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
