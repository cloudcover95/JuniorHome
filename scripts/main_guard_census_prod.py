#!/usr/bin/env python3
"""main_guard_census_prod.py - read-only census of script entry hygiene.

Parses every scripts/*.py with ast (never imports or executes them) and
reports, per file: module docstring present, `if __name__ == "__main__"`
guard present, argparse used, and whether it parses at all. Stdlib only,
no network, no writes. JuniorCloud LLC.
"""
import argparse
import ast
import json
import os
import sys


def has_main_guard(tree):
    for node in tree.body:
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__" and len(t.comparators) == 1):
                c = t.comparators[0]
                if isinstance(c, ast.Constant) and c.value == "__main__":
                    return True
    return False


def uses_argparse(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "argparse" for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == "argparse":
            return True
    return False


def inspect_file(path):
    row = {"file": os.path.basename(path), "parses": False,
           "docstring": False, "main_guard": False, "argparse": False}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        tree = ast.parse(src, filename=path)
    except (SyntaxError, ValueError, OSError):
        return row
    row["parses"] = True
    row["docstring"] = ast.get_docstring(tree) is not None
    row["main_guard"] = has_main_guard(tree)
    row["argparse"] = uses_argparse(tree)
    return row


def census(root, prod_only):
    rows = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        if prod_only and not name.endswith("_prod.py"):
            continue
        p = os.path.join(root, name)
        if os.path.isfile(p):
            rows.append(inspect_file(p))
    keys = ("parses", "docstring", "main_guard", "argparse")
    summary = {"files": len(rows)}
    for k in keys:
        summary[k] = sum(1 for r in rows if r[k])
    return {"root": os.path.abspath(root), "summary": summary, "rows": rows}


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Read-only ast census of scripts/*.py entry hygiene "
                    "(docstring, __main__ guard, argparse). Never executes "
                    "the scanned files.")
    ap.add_argument("--root", default=here, help="folder to scan (default: this script's folder)")
    ap.add_argument("--prod-only", action="store_true", help="only *_prod.py files")
    ap.add_argument("--missing", choices=("docstring", "main_guard", "argparse", "parses"),
                    help="list only files missing this property")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.root):
        print("not a directory: %s" % a.root, file=sys.stderr)
        return 2
    res = census(a.root, a.prod_only)
    if a.missing:
        res["rows"] = [r for r in res["rows"] if not r[a.missing]]
    if a.json:
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    s = res["summary"]
    print("root %s" % res["root"])
    print("files %d  parses %d  docstring %d  main_guard %d  argparse %d" % (
        s["files"], s["parses"], s["docstring"], s["main_guard"], s["argparse"]))
    if a.missing:
        for r in res["rows"]:
            print("  missing %s: %s" % (a.missing, r["file"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
