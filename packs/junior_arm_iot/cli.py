"""CLI for the TritARM host. Stdlib."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import host

Machine = host.Machine
PROFILES = host.PROFILES

HERE = Path(__file__).resolve().parent
DEMO = HERE / "images" / "pad_orbit.tas"


def _machine(ns):
    m = Machine(profile=ns.profile, roster=ns.roster)
    src = ns.image if ns.image else str(DEMO)
    m.load_path(src)
    return m


def cmd_demo(ns):
    m = _machine(ns)
    snap = m.run(ns.steps)
    json.dump(snap, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_run(ns):
    return cmd_demo(ns)


def cmd_profiles(_):
    for name, p in PROFILES.items():
        print("%s  ram=%d  W~%s  %s" % (name, p.ram, p.watts_hint, p.notes))
    return 0


def cmd_serve(ns):
    m = _machine(ns)
    m.run(ns.steps)

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/intent"):
                m.run(64)
                body = (m.intent_line() + "\n").encode()
                self._send(200, body, "application/json")
                return
            if self.path.startswith("/snap"):
                m.run(16)
                body = json.dumps(m.snapshot()).encode()
                self._send(200, body, "application/json")
                return
            page = (
                "<!doctype html><meta charset=utf-8><title>junior_arm_iot</title>"
                "<pre id=p></pre><script>"
                "async function tick(){const r=await fetch('/intent');"
                "document.getElementById('p').textContent=await r.text()}"
                "tick();setInterval(tick,250)</script>"
            )
            self._send(200, page.encode(), "text/html; charset=utf-8")

    httpd = ThreadingHTTPServer((ns.bind, ns.port), H)
    print("junior_arm_iot intent on http://%s:%d  (not a nintendo product)" % (ns.bind, ns.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="junior_arm_iot")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", default="generic_iot")
    common.add_argument("--roster", default="Forge")
    common.add_argument("--image", default="")
    common.add_argument("--steps", type=int, default=4000)
    p.add_argument("--profile", default="generic_iot")
    p.add_argument("--roster", default="Forge")
    p.add_argument("--image", default="")
    p.add_argument("--steps", type=int, default=4000)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("demo", parents=[common])
    sub.add_parser("run", parents=[common])
    sub.add_parser("profiles", parents=[common])
    s = sub.add_parser("serve-intent", parents=[common])
    s.add_argument("--bind", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8777)
    ns = p.parse_args(argv)
    if ns.cmd in (None, "demo"):
        return cmd_demo(ns)
    if ns.cmd == "run":
        return cmd_run(ns)
    if ns.cmd == "profiles":
        return cmd_profiles(ns)
    if ns.cmd == "serve-intent":
        return cmd_serve(ns)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
