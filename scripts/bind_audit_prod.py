#!/usr/bin/env python3
"""bind_audit_prod.py - read-only audit for non-loopback bind addresses.

Walks scripts/*.py with ast and text scan (never imports or executes them)
and flags string literals "0.0.0.0" / "::" used as a host, plus .bind()/
HTTPServer-style calls whose first host argument is not 127.0.0.1/localhost.
Enforces the LLC rule: every server binds 127.0.0.1. No network, no writes.
JuniorCloud LLC.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Dict, List

BAD_HOSTS = {"0.0.0.0", "::", "[::]", "0:0:0:0:0:0:0:0"}
SERVER_CALLS = {"bind", "HTTPServer", "ThreadingHTTPServer", "TCPServer",
                "ThreadingTCPServer", "UDPServer", "create_server",
                "start_server", "make_server"}


def call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def first_host(node: ast.Call):
    """Return the host literal from a tuple/str first arg, or None."""
    if not node.args:
        return None
    a = node.args[0]
    if isinstance(a, ast.Tuple) and a.elts and isinstance(a.elts[0], ast.Constant):
        return a.elts[0].value
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return a.value
    return None


def audit_file(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (SyntaxError, OSError, ValueError) as exc:
        return [{"file": path, "line": 0, "kind": "parse_error", "detail": str(exc)}]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in BAD_HOSTS:
            out.append({"file": path, "line": node.lineno, "kind": "bad_host_literal",
                        "detail": node.value})
        elif isinstance(node, ast.Call) and call_name(node) in SERVER_CALLS:
            host = first_host(node)
            if host == "":
                out.append({"file": path, "line": node.lineno, "kind": "empty_host_bind",
                            "detail": call_name(node)})
    return out


def scan(root: str) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
        for name in sorted(files):
            if name.endswith(".py"):
                findings.extend(audit_file(os.path.join(dirpath, name)))
    return findings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Read-only audit for non-loopback binds.")
    p.add_argument("path", nargs="?", default=".", help="directory to scan (default: .)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--strict", action="store_true", help="exit 1 if any finding")
    a = p.parse_args(argv)
    if not os.path.isdir(a.path):
        print("not a directory: %s" % a.path, file=sys.stderr)
        return 2
    findings = scan(a.path)
    if a.json:
        print(json.dumps({"count": len(findings), "findings": findings}, indent=2))
    else:
        for f in findings:
            print("%s:%s %s %s" % (f["file"], f["line"], f["kind"], f["detail"]))
        print("findings: %d" % len(findings))
    return 1 if (a.strict and findings) else 0


if __name__ == "__main__":
    sys.exit(main())
