#!/usr/bin/env python3
"""additive_check_prod.py - confirm a change set is additive only.

JuniorCloud LLC. cloudcover95. Stdlib only. No network. No server.

Reads `git diff --name-status` style lines (from a file or stdin) and
checks them against the LLC builder guards:
  - additive only: every entry must be an add (A); no M, D, R, C, T
  - no .github/ paths (no GitHub Actions)
  - optional: at most N new files (guard 1: one new file per turn)
  - optional: new files must match an allowed glob (e.g. scripts/*_prod.py)

Exit 0 when the change set passes, 1 when it fails, 2 on usage error.

Example:
  git diff --name-status HEAD~1..HEAD | python3 scripts/additive_check_prod.py --max-new 1 --allow 'scripts/*_prod.py'
"""
import argparse
import fnmatch
import json
import sys

STATUS_NAMES = {"A": "add", "M": "modify", "D": "delete", "R": "rename",
                "C": "copy", "T": "type-change", "U": "unmerged", "X": "unknown"}


def parse_lines(lines):
    """Return list of (status_letter, [paths]) from name-status lines."""
    out = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
        if len(parts) < 2:
            out.append(("X", [line.strip()]))
            continue
        letter = parts[0].strip()[:1].upper() or "X"
        out.append((letter, [p.strip() for p in parts[1:] if p.strip()]))
    return out


def check(entries, max_new=None, allow=None):
    flags = []
    added = []
    for letter, paths in entries:
        for p in paths:
            if p == ".github" or p.startswith(".github/") or "/.github/" in p:
                flags.append("no-.github: %s" % p)
        if letter != "A":
            flags.append("non-additive %s: %s" % (STATUS_NAMES.get(letter, letter), " -> ".join(paths)))
            continue
        added.append(paths[0])
        if allow and not any(fnmatch.fnmatch(paths[0], g) for g in allow):
            flags.append("outside allowed globs: %s" % paths[0])
    if max_new is not None and len(added) > max_new:
        flags.append("too many new files: %d > %d" % (len(added), max_new))
    return {"ok": not flags, "added": added, "flags": flags,
            "entries": len(entries)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Check git name-status lines for additive-only, no .github, file-count and path guards.")
    ap.add_argument("file", nargs="?", default="-",
                    help="name-status file to read (default: stdin)")
    ap.add_argument("--max-new", type=int, default=None,
                    help="maximum number of new files allowed")
    ap.add_argument("--allow", action="append", default=None,
                    help="glob a new file must match (repeatable)")
    ap.add_argument("--json", action="store_true", help="print JSON result")
    args = ap.parse_args(argv)

    if args.max_new is not None and args.max_new < 0:
        ap.error("--max-new must be >= 0")
    try:
        if args.file == "-":
            lines = sys.stdin.readlines()
        else:
            with open(args.file, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    result = check(parse_lines(lines), args.max_new, args.allow)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for a in result["added"]:
            print("add  %s" % a)
        for f in result["flags"]:
            print("FLAG %s" % f)
        print("ok" if result["ok"] else "fail")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
