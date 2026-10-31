"""Guard 4 census for scripts/*_prod.py.

LLC guard 4 says: mint false, socket false, layer_one false, fetch false,
ue5_launch false. This tool reads Python files (default: scripts/*_prod.py
beside this file) and reports every place one of those five flags is given a
literal value, as a variable assignment, a dict key, or a keyword argument.
Any literal True is flagged. Read-only. Standard library only. No network.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

FLAGS = ("mint", "socket", "layer_one", "fetch", "ue5_launch")
HERE = Path(__file__).resolve().parent


def _lit(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def scan_source(text, filename="<src>"):
    """Return list of hits: {line, flag, value, kind}."""
    tree = ast.parse(text, filename=filename)
    hits = []

    def add(name, val, line, kind):
        if name in FLAGS and val is not None:
            hits.append({"line": line, "flag": name, "value": val, "kind": kind})

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add(_name(t), _lit(node.value), node.lineno, "assign")
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            add(_name(node.target), _lit(node.value), node.lineno, "assign")
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    add(_name(k), _lit(v), getattr(k, "lineno", node.lineno), "dict")
        elif isinstance(node, ast.keyword) and node.arg:
            add(node.arg, _lit(node.value), node.value.lineno, "kwarg")
    hits.sort(key=lambda h: (h["line"], h["flag"]))
    return hits


def scan_file(path):
    rec = {"file": str(path), "ok": True, "hits": [], "error": None}
    try:
        rec["hits"] = scan_source(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, UnicodeDecodeError, OSError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["ok"] = not any(h["value"] is True for h in rec["hits"])
    return rec


def collect(paths, pattern):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            files.append(p)
    return files


def selftest():
    src = (
        "mint = False\n"
        "cfg = {'socket': False, 'fetch': True}\n"
        "run(layer_one=False, ue5_launch=True)\n"
        "other = True\n"
    )
    hits = scan_source(src)
    bad = sorted(h["flag"] for h in hits if h["value"] is True)
    good = sorted(h["flag"] for h in hits if h["value"] is False)
    assert bad == ["fetch", "ue5_launch"], bad
    assert good == ["layer_one", "mint", "socket"], good
    return {"selftest": "ok", "hits": len(hits)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=[str(HERE)],
                    help="files or folders to scan (default: this scripts/ folder)")
    ap.add_argument("--glob", default="*_prod.py", help="pattern inside folders")
    ap.add_argument("--text", action="store_true", help="text output instead of JSON")
    ap.add_argument("--all", action="store_true", help="also list files with no hits")
    ap.add_argument("--selftest", action="store_true", help="run built-in check and exit")
    a = ap.parse_args(argv)

    if a.selftest:
        print(json.dumps(selftest()))
        return 0

    recs = [scan_file(f) for f in collect(a.paths, a.glob)]
    flagged = [r for r in recs if not r["ok"] or r["error"]]
    shown = recs if a.all else [r for r in recs if r["hits"] or r["error"]]
    summary = {"guard": 4, "flags": list(FLAGS), "files": len(recs),
               "with_hits": sum(1 for r in recs if r["hits"]),
               "true_flags": sum(1 for r in recs if not r["ok"]),
               "errors": sum(1 for r in recs if r["error"]),
               "ok": not flagged}
    if a.text:
        for r in shown:
            if r["error"]:
                print(f"ERR  {r['file']}: {r['error']}")
                continue
            for h in r["hits"]:
                tag = "TRUE" if h["value"] else "ok  "
                print(f"{tag} {r['file']}:{h['line']} {h['flag']}={h['value']} ({h['kind']})")
        print(json.dumps(summary))
    else:
        print(json.dumps({"summary": summary, "files": shown}, indent=2))
    return 0 if summary["true_flags"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
