#!/usr/bin/env python3
"""lane_window_prod.py - LLC lane window check for JuniorHome.

Reports which LLC bot lane owns a given America/Denver time, whether LLC
writes are allowed (02:00-04:45 MT), and whether the personal quant team
window (05:05-07:05 MT) is active. Read-only: never writes, never fetches.

Guard flags: mint false, socket false, layer_one false, fetch false,
ue5_launch false. Stdlib only. Optional --serve binds 127.0.0.1 only.
JuniorCloud LLC. cloudcover95. Additive only.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys

try:
    from zoneinfo import ZoneInfo
    MT = ZoneInfo("America/Denver")
except Exception:  # pragma: no cover - tzdata missing
    MT = None

FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
RATE = 0.05
HOST = "127.0.0.1"

ROSTER = [
    ("02:00", "Chief of Staff", ["reads only"]),
    ("02:15", "JuniorHome Builder", ["JuniorHome", "JuniorDrive", "JuniorFetch"]),
    ("02:45", "JuniorOS Builder", ["JuniorLLM", "JuniorPiThon", "JuniorPiPython"]),
    ("03:15", "Engine Math", ["web3node", "JuniorQuant", "stocksnode",
                              "JuniorStock", "JuniorSOL"]),
    ("03:45", "Suite Scripts", ["JuniorPython-Suite", "JuniorEngrTools",
                                "JuniorHome scripts"]),
    ("04:15", "Spatial Suite", ["FrameForge", "FrameForge2D", "JuniorOmega",
                                "JuniorClimbs", "JuniorCoach", "JuniorPoker"]),
    ("04:45", "Inference Core", ["BitNet-mlx", "JuniorMemSys-Suite",
                                 "crispy-mouse", "JuniorAGI_SDK", "AGI_SDK"]),
]
LLC_WINDOW = ("02:00", "04:45")
QUANT_WINDOW = ("05:05", "07:05")


def _mins(hhmm):
    h, m = hhmm.split(":")
    h, m = int(h), int(m)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError("bad time: %s" % hhmm)
    return h * 60 + m


def _in(t, window):
    return _mins(window[0]) <= t <= _mins(window[1])


def close_rule(prev, pack, text):
    """sha256(prev || pack || text)[:16]"""
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def check(hhmm):
    t = _mins(hhmm)
    lane = None
    for start, bot, repos in ROSTER:
        if t >= _mins(start):
            lane = {"bot": bot, "start": start, "repos": repos}
    llc = _in(t, LLC_WINDOW)
    quant = _in(t, QUANT_WINDOW)
    if not llc:
        lane = None
    return {
        "time_mt": hhmm,
        "llc_write_ok": llc and not quant,
        "quant_window": quant,
        "lane": lane,
        "rate": RATE,
        "flags": FLAGS,
        "close": close_rule("0" * 16, "lane_window", hhmm),
    }


def now_hhmm():
    now = dt.datetime.now(MT) if MT else dt.datetime.now()
    return now.strftime("%H:%M")


def serve(port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(check(now_hhmm())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer((HOST, port), H)
    print("serving on http://%s:%d" % (HOST, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def selftest():
    assert check("03:45")["lane"]["bot"] == "Suite Scripts"
    assert check("03:45")["llc_write_ok"] is True
    assert check("04:46")["llc_write_ok"] is False
    assert check("05:30")["quant_window"] is True
    assert check("05:30")["lane"] is None
    assert check("01:59")["lane"] is None
    assert len(close_rule("a", "b", "c")) == 16
    assert not any(FLAGS.values())
    return "ok"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--at", help="HH:MM in America/Denver (default: now)")
    p.add_argument("--selftest", action="store_true", help="run checks")
    p.add_argument("--serve", type=int, metavar="PORT",
                   help="serve JSON on 127.0.0.1:PORT")
    a = p.parse_args(argv)
    if a.selftest:
        print(selftest())
        return 0
    if a.serve:
        serve(a.serve)
        return 0
    print(json.dumps(check(a.at or now_hhmm()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
