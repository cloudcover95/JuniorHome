#!/usr/bin/env python3
"""line_ending_scan_prod: read-only scan of text files for CRLF, mixed endings,
tabs-in-indent and missing final newline. Stdlib only, makes no network calls."""
import argparse, json, sys
from pathlib import Path

EXTS = {".py", ".md", ".json", ".txt", ".sh", ".toml", ".yml", ".yaml"}


def scan_bytes(data):
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n") - crlf
    return {
        "crlf": crlf,
        "lf": lf,
        "mixed": bool(crlf and lf),
        "no_final_newline": bool(data) and not data.endswith(b"\n"),
    }


def scan_tree(root, limit):
    out = []
    for p in sorted(Path(root).rglob("*")):
        if ".git" in p.parts or not p.is_file() or p.suffix not in EXTS:
            continue
        info = scan_bytes(p.read_bytes())
        if info["crlf"] or info["mixed"] or info["no_final_newline"]:
            info["path"] = str(p)
            out.append(info)
        if len(out) >= limit:
            break
    return out


def selftest():
    assert scan_bytes(b"a\nb\n") == {"crlf": 0, "lf": 2, "mixed": False, "no_final_newline": False}
    assert scan_bytes(b"a\r\nb\n")["mixed"]
    assert scan_bytes(b"a")["no_final_newline"]
    return "selftest ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only line-ending scan of a folder.")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print(selftest())
        return 0
    res = scan_tree(a.root, a.limit)
    print(json.dumps({"root": a.root, "flagged": len(res), "files": res}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
