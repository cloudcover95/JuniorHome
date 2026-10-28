#!/usr/bin/env python3
"""guard_digest_prod.py - read-only digest of docs/LLC_GUARDS.md.

Parses the LLC guard sheet and prints the numbered builder and chief
guards plus a sha256 of the sheet, so every bot in the roster can
confirm it is reading the same version. Never writes to docs/.

Stdlib only. No network. No server. Additive. JuniorCloud LLC.
"""
import argparse
import hashlib
import json
import os
import re
import sys

DEFAULT_SHEET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "LLC_GUARDS.md")

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
ITEM_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")


def parse(text):
    """Return {section_name: [guard text, ...]} for numbered lists."""
    sections = {}
    current = None
    for line in text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            continue
        m = ITEM_RE.match(line)
        if m and current:
            sections.setdefault(current, []).append(m.group(2))
    return sections


def digest(text):
    sections = parse(text)
    builder = next((v for k, v in sections.items() if "every builder" in k.lower()), [])
    chief = next((v for k, v in sections.items() if k.lower().startswith("chief")), [])
    return {
        "sha256_16": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "builder_guards": builder,
        "builder_count": len(builder),
        "chief_guards": chief,
        "chief_count": len(chief),
        "binds_loopback": any("127.0.0.1" in g for g in builder),
        "ok": len(builder) == 10,
    }


SAMPLE = """# LLC bot guards
## Guards (every builder)
1. One repo per turn.
2. Additive only.
3. Bind 127.0.0.1. No 0.0.0.0.
4. a
5. b
6. c
7. d
8. No model pull.
9. If the tip is recent, take a different path.
10. If the turn drops, stop.
## Chief guards
1. Read the tips.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only digest of docs/LLC_GUARDS.md (builder and chief guards, sha256).")
    ap.add_argument("--sheet", default=DEFAULT_SHEET, help="path to LLC_GUARDS.md")
    ap.add_argument("--expect", help="expected sha256_16; exit 1 if it differs")
    ap.add_argument("--smoke", action="store_true", help="run on a built-in sample sheet")
    ap.add_argument("--json", action="store_true", help="print JSON")
    args = ap.parse_args(argv)

    if args.smoke:
        text = SAMPLE
    else:
        try:
            with open(args.sheet, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print("guard_digest: cannot read %s: %s" % (args.sheet, exc), file=sys.stderr)
            return 2
    d = digest(text)
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print("sheet sha256_16 %s" % d["sha256_16"])
        print("builder guards %d, chief guards %d, loopback %s" % (
            d["builder_count"], d["chief_count"], d["binds_loopback"]))
        for i, g in enumerate(d["builder_guards"], 1):
            print("  %2d. %s" % (i, g))
        print("ok" if d["ok"] else "FLAG: expected 10 builder guards")
    if args.expect and args.expect != d["sha256_16"]:
        print("FLAG: sheet hash %s != expected %s" % (d["sha256_16"], args.expect), file=sys.stderr)
        return 1
    return 0 if d["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
