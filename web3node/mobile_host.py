"""Local mobile page. Bind 127.0.0.1 only. No store upload."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
PAGE = b"<!DOCTYPE html><meta name=viewport content='width=device-width'><title>JuniorHome</title><p>local</p>"


def ticket() -> dict:
    body = {
        "host": "mobile",
        "bind": "127.0.0.1",
        "port": 8766,
        "store": False,
        "open_source": True,
        "supports": ["usb-sound", "usb-hid", "sbc", "mobile-browser"],
        "model_pull": False,
    }
    OS.mkdir(parents=True, exist_ok=True)
    (OS / "mobile.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body


def serve() -> None:
    ticket()

    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)

        def log_message(self, fmt: str, *args) -> None:
            return

    ThreadingHTTPServer(("127.0.0.1", 8766), H).serve_forever()
