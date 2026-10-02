"""Serve Home on 127.0.0.1. app and ui only."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web3node"))
from stack_tick import tick
ALLOW = ("app", "ui")
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/app", "/app/"):
            return self._file(ROOT / "app" / "index.html", "text/html")
        if self.path == "/status":
            return self._json({"bind": "127.0.0.1", "admit": False, "launch": False, "stack": "/stack"})
        if self.path == "/stack":
            return self._json(tick())
        parts = [p for p in self.path.split("/") if p]
        if not parts or parts[0] not in ALLOW or ".." in parts:
            self.send_error(404)
            return
        path = ROOT.joinpath(*parts)
        if not path.is_file():
            self.send_error(404)
            return
        self._file(path, "text/html" if path.suffix == ".html" else "text/plain")
    def _json(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(raw)
    def _file(self, path, kind):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", kind)
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, fmt, *args):
        return
if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8766), H).serve_forever()
