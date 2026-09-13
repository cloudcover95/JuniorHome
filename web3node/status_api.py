"""Stdlib GET /status → latch JSON. Not Express."""
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from latch import latch

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") not in ("", "/status", "/latch"):
            self.send_error(404); return
        body = json.dumps(latch(), default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, fmt, *args):
        return

if __name__ == "__main__":
    print(json.dumps({"listen": "127.0.0.1:8765", "clone": False}))
