"""Serve Home on 127.0.0.1. app and ui only."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ALLOW = ("app", "ui")
STATUS = {"bind": "127.0.0.1", "admit": False, "launch": False,
          "ports": ["JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian"],
          "modules": ["gaia", "deck"], "opened": False, "model_pull": False}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/app", "/app/"):
            return self._file(ROOT / "app" / "index.html", "text/html")
        if self.path == "/status":
            body = json.dumps(STATUS).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        parts = [p for p in self.path.split("/") if p]
        if not parts or parts[0] not in ALLOW or ".." in parts:
            self.send_error(404)
            return
        path = ROOT.joinpath(*parts)
        if not path.is_file():
            self.send_error(404)
            return
        self._file(path, "text/html" if path.suffix == ".html" else "text/plain")
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
