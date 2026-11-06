#!/usr/bin/env python3
"""import_stdlib_audit_prod.py - read-only AST audit of non-stdlib imports.

Walks scripts/*.py with ast (never imports or executes them) and reports
top-level and nested import roots that are not in the Python standard
library. Useful for enforcing the LLC stdlib-only rule on prod scripts.
No network, no writes. JuniorCloud LLC.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Iterable, List, Set


# Frozen snapshot of common stdlib top-level names for CPython 3.10–3.13.
# Intentionally explicit (no importlib.metadata) so this script stays offline.
STDLIB_ROOTS: Set[str] = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii", "binhex",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath",
    "cmd", "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
    "email", "encodings", "enum", "errno", "faulthandler", "fcntl",
    "filecmp", "fileinput", "fnmatch", "fractions", "ftplib", "functools",
    "gc", "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
    "hashlib", "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "imp", "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder",
    "multiprocessing", "netrc", "nis", "nntplib", "numbers", "operator",
    "optparse", "os", "ossaudiodev", "pathlib", "pdb", "pickle", "pickletools",
    "pipes", "pkgutil", "platform", "plistlib", "poplib", "posix", "posixpath",
    "pprint", "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd",
    "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl",
    "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile",
    "telnetlib", "tempfile", "termios", "test", "textwrap", "threading",
    "time", "timeit", "tkinter", "token", "tokenize", "tomllib", "trace",
    "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types",
    "typing", "unicodedata", "unittest", "urllib", "uu", "uuid", "venv",
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound",
    "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport",
    "zlib", "_thread", "__future__",
}


def root_of(name: str) -> str:
    return name.split(".", 1)[0] if name else ""


def collect_imports(tree: ast.AST) -> Set[str]:
    roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                r = root_of(alias.name)
                if r:
                    roots.add(r)
        elif isinstance(node, ast.ImportFrom):
            # relative imports (level > 0) are local package refs, skip as stdlib check
            if node.level and not node.module:
                continue
            if node.module:
                r = root_of(node.module)
                if r and node.level == 0:
                    roots.add(r)
    return roots


def inspect_file(path: str) -> dict:
    row = {
        "file": os.path.basename(path),
        "parses": False,
        "imports": [],
        "non_stdlib": [],
        "ok": False,
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        tree = ast.parse(src, filename=path)
    except (SyntaxError, ValueError, OSError):
        return row
    roots = sorted(collect_imports(tree))
    non = sorted(r for r in roots if r not in STDLIB_ROOTS)
    row["parses"] = True
    row["imports"] = roots
    row["non_stdlib"] = non
    row["ok"] = len(non) == 0
    return row


def audit(root: str, prod_only: bool) -> dict:
    rows: List[dict] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        if prod_only and not name.endswith("_prod.py"):
            continue
        p = os.path.join(root, name)
        if os.path.isfile(p):
            rows.append(inspect_file(p))
    summary = {
        "files": len(rows),
        "parses": sum(1 for r in rows if r["parses"]),
        "stdlib_ok": sum(1 for r in rows if r["ok"]),
        "non_stdlib": sum(1 for r in rows if r["parses"] and not r["ok"]),
        "parse_fail": sum(1 for r in rows if not r["parses"]),
    }
    return {"root": os.path.abspath(root), "summary": summary, "rows": rows}


def main(argv: Iterable[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Read-only AST audit of non-stdlib imports in scripts/*.py. "
                    "Never executes the scanned files.")
    ap.add_argument("--root", default=here,
                    help="folder to scan (default: this script's folder)")
    ap.add_argument("--prod-only", action="store_true",
                    help="only *_prod.py files")
    ap.add_argument("--violations", action="store_true",
                    help="list only files with non-stdlib imports")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not os.path.isdir(a.root):
        print("not a directory: %s" % a.root, file=sys.stderr)
        return 2
    res = audit(a.root, a.prod_only)
    if a.violations:
        res["rows"] = [r for r in res["rows"] if r["parses"] and not r["ok"]]
    if a.json:
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    s = res["summary"]
    print("root %s" % res["root"])
    print("files %d  parses %d  stdlib_ok %d  non_stdlib %d  parse_fail %d" % (
        s["files"], s["parses"], s["stdlib_ok"], s["non_stdlib"], s["parse_fail"]))
    if a.violations:
        for r in res["rows"]:
            print("  %s: %s" % (r["file"], ", ".join(r["non_stdlib"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
