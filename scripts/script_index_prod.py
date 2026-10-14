"""Index scripts/: list each script with the first line of its module docstring.

Read-only. Standard library only. Emits a text table (default), JSON, or Markdown.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(path):
    rec = {"file": path.name, "summary": "", "error": None}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        doc = ast.get_docstring(tree) or ""
        rec["summary"] = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(HERE), help="directory to index (default: scripts/)")
    p.add_argument("--pattern", default="*.py", help="glob pattern (default: *.py)")
    p.add_argument("--format", choices=("text", "json", "md"), default="text")
    p.add_argument("--missing", action="store_true", help="only list files without a docstring")
    a = p.parse_args(argv)
    recs = [summarize(f) for f in sorted(Path(a.dir).glob(a.pattern)) if f.is_file()]
    if a.missing:
        recs = [r for r in recs if not r["summary"]]
    if a.format == "json":
        print(json.dumps(recs, indent=2))
    elif a.format == "md":
        print("| script | summary |\n|---|---|")
        for r in recs:
            print(f"| `{r['file']}` | {(r['summary'] or r['error'] or '').replace('|', '/')} |")
    else:
        w = max((len(r["file"]) for r in recs), default=4)
        for r in recs:
            print(f"{r['file']:<{w}}  {r['summary'] or ('!! ' + r['error'] if r['error'] else '-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
