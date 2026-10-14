"""Audit scripts/*_prod.py: syntax, __main__ guard, argparse use, and local imports.

Read-only. Standard library only. Emits JSON (default) or a text table.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audit_file(path):
    rec = {"file": path.name, "ok": True, "main_guard": False,
           "argparse": False, "imports": [], "error": None}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"):
                rec["main_guard"] = True
    rec["argparse"] = "argparse" in mods
    std = getattr(sys, "stdlib_module_names", set())
    rec["imports"] = sorted(mods)
    rec["non_stdlib"] = sorted(m for m in mods if std and m not in std)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(HERE), help="scripts directory")
    p.add_argument("--glob", default="*_prod.py")
    p.add_argument("--text", action="store_true", help="text table instead of JSON")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any file fails to parse or lacks a main guard")
    a = p.parse_args(argv)
    files = sorted(Path(a.dir).glob(a.glob))
    recs = [audit_file(f) for f in files]
    summary = {"total": len(recs),
               "parse_errors": sum(not r["ok"] for r in recs),
               "no_main_guard": sum(r["ok"] and not r["main_guard"] for r in recs),
               "with_argparse": sum(r["argparse"] for r in recs)}
    if a.text:
        for r in recs:
            flag = "ERR " if not r["ok"] else ("ok  " if r["main_guard"] else "noMG")
            extra = r["error"] or ",".join(r.get("non_stdlib", []))
            print(f"{flag} {r['file']:<40} {extra}")
        print(json.dumps(summary))
    else:
        print(json.dumps({"summary": summary, "files": recs}, indent=2))
    if a.strict and (summary["parse_errors"] or summary["no_main_guard"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
