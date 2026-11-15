#!/usr/bin/env python3
"""docstring_header_audit_prod: read-only audit of *_prod.py module headers.

Uses ast only (never imports or runs the scanned files). Reports files with a
missing module docstring, a missing shebang, or no argparse/main entrypoint.
Stdlib only. Opens no sockets. Exit 0 = clean, 1 = findings.
"""
import argparse
import ast
import glob
import json
import os
import sys


def check(path):
    problems = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        tree = ast.parse(src, filename=path)
    except (OSError, SyntaxError) as exc:
        return ["unparsable: %s" % exc]
    if not ast.get_docstring(tree):
        problems.append("no module docstring")
    if not src.startswith("#!"):
        problems.append("no shebang")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "main" not in funcs and "argparse" not in src and "__main__" not in src:
        problems.append("no entrypoint")
    del names
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit *_prod.py headers (read-only, ast only).")
    ap.add_argument("root", nargs="?", default="scripts", help="folder holding *_prod.py (default: scripts)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    a = ap.parse_args(argv)
    report = {}
    for path in sorted(glob.glob(os.path.join(a.root, "*_prod.py"))):
        p = check(path)
        if p:
            report[path] = p
    if a.json:
        print(json.dumps({"root": a.root, "findings": report}, indent=2))
    else:
        for k, v in report.items():
            print("%s: %s" % (k, "; ".join(v)))
        print("files with findings: %d" % len(report))
    return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
