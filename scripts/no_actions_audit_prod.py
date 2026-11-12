#!/usr/bin/env python3
"""no_actions_audit_prod.py - read-only audit for GitHub Actions / bot config.

Walks a checkout and flags any .github directory, workflow yml/yaml,
dependabot or copilot config. Stdlib only. Never writes, never networks.
Exit 0 = clean, 1 = findings, 2 = bad path.
JuniorCloud LLC. cloudcover95. Additive only.
"""
import argparse
import json
import os
import sys

SKIP = {".git", "node_modules", "__pycache__"}
NAMES = {"dependabot.yml", "dependabot.yaml", "copilot-instructions.md",
         "renovate.json"}


def audit(root):
    findings = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        rel = os.path.relpath(cur, root)
        if os.path.basename(cur) == ".github":
            findings.append((rel, "github-dir"))
        if rel.replace(os.sep, "/").endswith(".github/workflows"):
            for f in files:
                if f.lower().endswith((".yml", ".yaml")):
                    findings.append((os.path.join(rel, f), "workflow"))
        for f in files:
            if f.lower() in NAMES:
                findings.append((os.path.join(rel, f), "bot-config"))
    return sorted(set(findings))


def main(argv=None):
    p = argparse.ArgumentParser(description="Read-only audit: no GitHub Actions, "
                                "Dependabot or Copilot config in a checkout.")
    p.add_argument("path", nargs="?", default=".", help="checkout root (default .)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    a = p.parse_args(argv)
    if not os.path.isdir(a.path):
        print("not a directory: %s" % a.path, file=sys.stderr)
        return 2
    res = audit(a.path)
    if a.json:
        print(json.dumps({"path": a.path, "clean": not res,
                          "findings": [{"path": x, "kind": k} for x, k in res]}))
    else:
        for x, k in res:
            print("%s\t%s" % (k, x))
        print("clean" if not res else "%d finding(s)" % len(res))
    return 1 if res else 0


if __name__ == "__main__":
    sys.exit(main())
