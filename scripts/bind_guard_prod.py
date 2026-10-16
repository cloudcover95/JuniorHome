"""Scan scripts for bind/listen hosts that are not loopback (127.0.0.1 / localhost).

Read-only. Standard library only. Emits JSON (default) or a text table.
Flags literal host strings such as 0.0.0.0, ::, and other non-loopback addresses
passed to common bind/listen call patterns.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

LOOPBACK = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "127.0.0.1:0",
})

# Host literals that clearly expose a non-loopback listener.
BAD_HOSTS = frozenset({
    "0.0.0.0",
    "::",
    "[::]",
    "*",
})

HOST_RE = re.compile(
    r"""(?P<q>['"])(?P<host>0\.0\.0\.0|::|\[::\]|\*|localhost|127\.0\.0\.1|::1)(?P=q)"""
)

BIND_CALL_NAMES = frozenset({
    "bind",
    "listen",
    "serve_forever",
    "HTTPServer",
    "ThreadingHTTPServer",
    "TCPServer",
    "UDPServer",
    "socketserver",
})


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _looks_like_host(s):
    if not s or len(s) > 128:
        return False
    if s in BAD_HOSTS or s in LOOPBACK:
        return True
    # bare IPv4 / IPv6-ish or host:port
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", s):
        return True
    if re.fullmatch(r"\[[0-9a-fA-F:]+\]", s) or re.fullmatch(r"[0-9a-fA-F:]+", s):
        if ":" in s:
            return True
    return False


def _is_loopback_host(s):
    if s in LOOPBACK:
        return True
    if s.startswith("127."):
        return True
    base = s.split("%")[0].strip("[]")
    if base in ("127.0.0.1", "::1", "localhost"):
        return True
    return False


def _collect_host_from_call(node):
    """Return (host_str, lineno) candidates from bind/listen-style calls."""
    out = []
    func = node.func
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    if name not in BIND_CALL_NAMES and name not in ("run", "serve"):
        # still inspect first arg if call name suggests bind
        pass

    interesting = name in BIND_CALL_NAMES or name in ("run", "serve", "start")

    def add_from(expr):
        s = _const_str(expr)
        if s is not None and _looks_like_host(s):
            out.append((s, expr.lineno))
        elif isinstance(expr, (ast.Tuple, ast.List)) and expr.elts:
            s0 = _const_str(expr.elts[0])
            if s0 is not None and _looks_like_host(s0):
                out.append((s0, expr.elts[0].lineno))

    if interesting:
        for arg in node.args[:2]:
            add_from(arg)
        for kw in node.keywords:
            if kw.arg in ("host", "hostname", "address", "addr", "bind", "server_address"):
                add_from(kw.value)
    else:
        # keyword-only scan for explicit host= on any call
        for kw in node.keywords:
            if kw.arg in ("host", "hostname", "bind"):
                add_from(kw.value)
    return out


def scan_file(path):
    rec = {
        "file": path.name,
        "ok": True,
        "findings": [],
        "error": None,
    }
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for host, lineno in _collect_host_from_call(node):
                severity = "ok" if _is_loopback_host(host) else "bad"
                if severity == "ok":
                    continue
                rec["findings"].append({
                    "line": lineno,
                    "host": host,
                    "severity": severity,
                    "via": "call",
                })
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s in BAD_HOSTS:
                rec["findings"].append({
                    "line": getattr(node, "lineno", 0),
                    "host": s,
                    "severity": "bad",
                    "via": "literal",
                })

    # Deduplicate by (line, host); prefer call over bare literal
    by_key = {}
    for f in rec["findings"]:
        key = (f["line"], f["host"])
        prev = by_key.get(key)
        if prev is None or (prev["via"] == "literal" and f["via"] == "call"):
            by_key[key] = f
    uniq = sorted(by_key.values(), key=lambda x: (x["line"], x["host"]))
    rec["findings"] = uniq
    rec["bad_count"] = sum(1 for f in uniq if f["severity"] == "bad")
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(HERE), help="scripts directory to scan")
    p.add_argument("--glob", default="*_prod.py", help="file glob under --dir")
    p.add_argument("--text", action="store_true", help="text table instead of JSON")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any non-loopback bind host is found")
    a = p.parse_args(argv)
    root = Path(a.dir)
    if not root.is_dir():
        print(json.dumps({"error": f"not a directory: {root}"}), file=sys.stderr)
        return 2
    files = sorted(root.glob(a.glob))
    recs = [scan_file(f) for f in files]
    summary = {
        "total": len(recs),
        "parse_errors": sum(not r["ok"] for r in recs),
        "files_with_findings": sum(1 for r in recs if r.get("bad_count")),
        "bad_findings": sum(r.get("bad_count", 0) for r in recs),
    }
    if a.text:
        for r in recs:
            if not r["ok"]:
                print(f"ERR  {r['file']:<40} {r['error']}")
                continue
            if not r["findings"]:
                print(f"ok   {r['file']:<40}")
                continue
            for f in r["findings"]:
                print(f"BAD  {r['file']:<40} L{f['line']:<5} host={f['host']} ({f['via']})")
        print(json.dumps(summary))
    else:
        print(json.dumps({"summary": summary, "files": recs}, indent=2))
    if a.strict and (summary["parse_errors"] or summary["bad_findings"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
