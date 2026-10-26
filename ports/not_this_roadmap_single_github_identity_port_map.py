#!/usr/bin/env python3
"""Not this roadmap: "second GitHub bot user" single-identity port map.

Advances JuniorHome docs/ROADMAP.md named item (section "Not this roadmap"):
    0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user.
Sub-check: "second GitHub bot user". ports/not_this_roadmap_saas_bind_port_map.py
covers only the "0.0.0.0 SaaS" sub-check, and the scripts/ / web3node/
advance_* files for this sub-check only assert a typed SECOND_BOT = False.
This port map reads the identity surfaces in this clone, located from this
file's own path, and checks they all resolve to the one account cloudcover95:

  Home      ROADMAP heading + item verbatim; .git/config origin owner and repo
            [user] identity; catalog/repos.json owner + profile repo; every
            github.com link to a catalog repo is under cloudcover95.
  OSai      no git identity override (git config user.*, GIT_AUTHOR_*/
            GIT_COMMITTER_*) in code, none at all in deploy files; no GitHub
            token literal (ghp_/gho_/ghu_/ghs_/ghr_/github_pat_/x-access-token)
            anywhere; no second-account token env name in code.
  JuniorLLM docs/ledger/*.md "authored as" handles and every
            *@users.noreply.github.com address are cloudcover95; role labels
            ("authored by the LLC account", "self-labelled") are advisories.
  Flagstaff docs/ZERO_TRUST.md Identity line ("GitHub connector = cloudcover95
            ... Not a second bot user.") and the deploy-identity fail rows in
            ZERO_TRUST.md / TECH_AUDIT.md; every "Owner: X." in docs is
            cloudcover95.
  FieldCore every scanned file decodes; .github workflow secrets (if a
            .github tree is in the clone) name only GITHUB_TOKEN; PERSONAL_BOT_TECH
            keeps "cloudcover95 stays the owner".

Verdicts: PASS (trit 1), ADVISORY (trit 0), FAIL (trit -1). Exit 1 on any FAIL.
Text reads only: no import of scanned code, no git subprocess, no sockets, no
network, no fetch, no model pull, no UE boot. Writes nothing in the repo
(--selftest builds fixtures under a temp dir).

Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false. Rate is a constant; no poll sets it.
Close rule: sha256(prev || pack || text)[:16], balanced trits packed 2-bit
(0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.
Usage: python3 ports/not_this_roadmap_single_github_identity_port_map.py [--selftest]
"""
from __future__ import annotations

import configparser
import hashlib
import json
import os
import re
import sys
import tempfile

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
SECOND_BOT = False
BIND = "127.0.0.1"
OWNER = "cloudcover95"
OWNER_EMAIL = "cloudcover95@users.noreply.github.com"
SIDES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ITEM_HEADING = "Not this roadmap"
ITEM = "0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user."
SUBCHECK = "second GitHub bot user"
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
SELF = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(SELF))

ZT_IDENTITY = ("GitHub connector = cloudcover95", "Not a second bot user.")
ZT_FAIL_ROW = "git config user.email rewrite in deploy scripts"
TA_FAIL_ROW = "git config identity override in deploy scripts"
PERSONAL_ROW = "cloudcover95 stays the owner"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
BIN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".gguf",
           ".bin", ".so", ".dylib", ".o", ".a", ".wasm", ".pyc", ".blend", ".stl"}
CODE_EXT = {".py", ".sh", ".zsh", ".bash", ".yml", ".yaml", ".toml", ".cfg", ".ini",
            ".service", ".json", ".rules", ".lua", ".swift", ".tsx", ".ts", ".js"}
CODE_NAMES = {"Makefile", "Dockerfile"}
DEPLOY_PREFIX = ("deployment/", "Makefile", "Dockerfile", "docker-compose")
MAX_BYTES = 2_000_000

GH_LINK = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
GIT_ID = [
    re.compile(r"git\s+(?:-c\s+|config\s+(?:--\w+\s+)*)user\.(name|email)[\s=]+[\"']?([^\s\"';]+)"),
    re.compile(r"[\"']config[\"']\s*,\s*(?:[\"']--\w+[\"']\s*,\s*)*[\"']user\.(name|email)[\"']\s*,\s*[\"']([^\"']+)"),
    re.compile(r"\bGIT_(?:AUTHOR|COMMITTER)_(NAME|EMAIL)\b[\"']?\s*[]:=,)]*\s*[\"']([^\"']+)"),
]
TOKEN_LIT = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
                       r"|x-access-token:[^@\s]+@")
TOKEN_ENV = re.compile(r"\b(GH_TOKEN|GH_ENTERPRISE_TOKEN|GITHUB_PAT\w*|\w*BOT_GITHUB\w*|GITHUB_BOT\w*)\b")
NOREPLY = re.compile(r"([A-Za-z0-9_.+-]+)@users\.noreply\.github\.com")
AUTHORED_AS = re.compile(r"authored as `?([A-Za-z0-9_-]+)")
ROLE_LABELS = ("authored by the LLC account", "self-labelled")
OWNER_DECL = re.compile(r"\bOwner:?\s+\**([A-Za-z0-9_-]+)\**[.,;]")
WF_SECRET = re.compile(r"secrets\.([A-Za-z0-9_]+)")
_ENC = {0: 0, 1: 1, -1: 2}
PASS, ADV, FAIL = 1, 0, -1


def pack(trits) -> bytes:
    """Balanced trits as 2-bit codes, MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def _read(root: str, rel: str) -> str:
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _is_code(rel: str) -> bool:
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1]
    return (ext in CODE_EXT or base in CODE_NAMES or base.startswith("Dockerfile")
            or (rel.startswith("scripts/") and not ext))


def walk(root: str) -> tuple[dict[str, str], list[str]]:
    texts, bad = {}, []
    for d, subs, files in os.walk(root):
        subs[:] = sorted(s for s in subs if s not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(d, f)
            if os.path.abspath(p) == SELF or os.path.splitext(f)[1].lower() in BIN_EXT:
                continue
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            try:
                if os.path.getsize(p) > MAX_BYTES:
                    continue
                with open(p, encoding="utf-8") as fh:
                    texts[rel] = fh.read()
            except UnicodeDecodeError:
                bad.append(rel)
            except OSError:
                bad.append(rel)
    return texts, bad


def git_config(root: str) -> dict | None:
    path = os.path.join(root, ".git", "config")
    if not os.path.isfile(path):
        return None
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error:
        return {"error": True}
    out = {"url": cp.get('remote "origin"', "url", fallback=""),
           "name": cp.get("user", "name", fallback=None),
           "email": cp.get("user", "email", fallback=None)}
    return out


def _ok_identity(kind: str, val: str) -> bool:
    val = val.strip().strip("<>")
    if val.startswith("$") or val.startswith("{"):
        return True  # templated from the environment, not a literal second identity
    if kind.lower() == "name":
        return val == OWNER
    m = NOREPLY.fullmatch(val)
    return bool(m) and m.group(1).split("+")[-1] == OWNER


def checks(root: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    add = lambda side, name, v: rows.append((side, name, v))  # noqa: E731
    ok = lambda b: PASS if b else FAIL  # noqa: E731
    texts, bad = walk(root)

    # Home
    rm = texts.get("docs/ROADMAP.md", "")
    sec = rm.split(f"## {ITEM_HEADING}", 1)
    add("Home", f'ROADMAP "## {ITEM_HEADING}" names "{ITEM}"', ok(len(sec) == 2 and ITEM in sec[1]))
    add("Home", f'ROADMAP sub-check "{SUBCHECK}" present', ok(len(sec) == 2 and SUBCHECK in sec[1]))
    gc = git_config(root)
    if gc is None:
        add("Home", ".git/config absent (not a clone); origin owner not checkable", ADV)
    elif gc.get("error"):
        add("Home", ".git/config unparsable", FAIL)
    else:
        m = GH_LINK.search(gc["url"])
        add("Home", f".git/config origin {gc['url'] or '(none)'} owner {OWNER}",
            ok(bool(m) and m.group(1) == OWNER) if gc["url"] else ADV)
        for kind in ("name", "email"):
            val = gc[kind]
            if val is None:
                add("Home", f".git/config user.{kind} unset (inherits global; not checked)", ADV)
            else:
                add("Home", f".git/config user.{kind} = {val}", ok(_ok_identity(kind, val)))
    try:
        cat = json.loads(texts.get("catalog/repos.json", "") or "null")
    except json.JSONDecodeError:
        cat = None
    names: set[str] = set()
    if not isinstance(cat, dict):
        add("Home", "catalog/repos.json readable", FAIL)
    else:
        names = {r.get("name") for r in cat.get("repos", []) if isinstance(r, dict)}
        add("Home", f"catalog/repos.json owner = {cat.get('owner')!r}", ok(cat.get("owner") == OWNER))
        meta = [r for r in cat.get("repos", []) if isinstance(r, dict) and r.get("role") == "meta"]
        add("Home", f"catalog profile repo is the single meta entry {OWNER}",
            ok(len(meta) == 1 and meta[0].get("name") == OWNER))
    first, strays, upstream = 0, [], set()
    names_l = {n.lower() for n in names if n} | {"juniorhome"}
    for rel, txt in texts.items():
        for m in GH_LINK.finditer(txt):
            own, repo = m.group(1), re.sub(r"\.git$", "", m.group(2)).rstrip(".")
            if own == OWNER:
                first += 1
            elif repo.lower() in names_l or own.lower().startswith(OWNER.lower()):
                strays.append(f"{rel}:{own}/{repo}")
            else:
                upstream.add(own)
    add("Home", f"{first} github.com links under {OWNER}; catalog repos under another owner: "
        f"{strays or 0}", ok(not strays))
    add("Home", f"{len(upstream)} upstream owners linked are non-catalog references only", PASS)

    # OSai
    overrides, deploy_hits, bad_ids = [], [], []
    for rel, txt in texts.items():
        if not _is_code(rel):
            continue
        for rx in GIT_ID:
            for m in rx.finditer(txt):
                line = txt.count("\n", 0, m.start()) + 1
                where = f"{rel}:{line}"
                overrides.append(where)
                if rel.startswith(DEPLOY_PREFIX):
                    deploy_hits.append(where)
                elif not _ok_identity(m.group(1), m.group(2)):
                    bad_ids.append(f"{where} {m.group(2)}")
    add("OSai", f"git identity overrides in code: {len(overrides)}; non-{OWNER}: {bad_ids or 0}",
        ok(not bad_ids))
    add("OSai", f"deploy files rewrite no git identity ({deploy_hits or 'none'})", ok(not deploy_hits))
    lits = sorted({rel for rel, txt in texts.items() if TOKEN_LIT.search(txt)})
    add("OSai", f"no GitHub token literal in {len(texts)} text files ({lits or 'none'})", ok(not lits))
    envs = sorted({f"{rel}:{m.group(1)}" for rel, txt in texts.items() if _is_code(rel)
                   for m in TOKEN_ENV.finditer(txt)})
    if not envs:
        add("OSai", "no second-account GitHub token env name in code", PASS)
    for e in envs:
        add("OSai", f"token env name {e} (whose login is not readable here)", ADV)

    # JuniorLLM ledger
    ledgers = sorted(r for r in texts if r.startswith("docs/ledger/") and r.endswith(".md"))
    handles = [(r, m.group(1)) for r in ledgers for m in AUTHORED_AS.finditer(texts[r])]
    others = sorted({f"{r}:{h}" for r, h in handles if h != OWNER})
    add("JuniorLLM", f"{len(ledgers)} ledgers, {len(handles)} 'authored as' handles all {OWNER} "
        f"({others or 'no other'})", ok(bool(ledgers) and not others) if ledgers else ADV)
    mails = [(r, m.group(1)) for r, txt in texts.items() for m in NOREPLY.finditer(txt)]
    odd = sorted({f"{r}:{u}" for r, u in mails if u.split("+")[-1] != OWNER})
    add("JuniorLLM", f"{len(mails)} noreply addresses all {OWNER} ({odd or 'no other'})", ok(not odd))
    for r in ledgers:
        for lab in ROLE_LABELS:
            if lab in texts[r]:
                add("JuniorLLM", f'{r} records "{lab}" (role label, not a handle; '
                    f"resolves to {OWNER} via ZERO_TRUST identity)", ADV)

    # Flagstaff
    zt = texts.get("docs/ZERO_TRUST.md", "")
    ident = zt.split("## Identity", 1)
    add("Flagstaff", f'ZERO_TRUST Identity: "{ZT_IDENTITY[0]}" ... "{ZT_IDENTITY[1]}"',
        ok(len(ident) == 2 and all(s in ident[1] for s in ZT_IDENTITY)))
    add("Flagstaff", f'ZERO_TRUST fail list names "{ZT_FAIL_ROW}"', ok(ZT_FAIL_ROW in zt))
    add("Flagstaff", f'TECH_AUDIT names "{TA_FAIL_ROW}"', ok(TA_FAIL_ROW in texts.get("docs/TECH_AUDIT.md", "")))
    decl = [(r, m.group(1)) for r, txt in texts.items() if r.endswith(".md") or r == "README.md"
            for m in OWNER_DECL.finditer(txt)]
    wrong = sorted({f"{r}:{o}" for r, o in decl if o != OWNER})
    add("Flagstaff", f"{len(decl)} 'Owner X.' declarations in docs all {OWNER} ({wrong or 'no other'})",
        ok(not wrong))
    add("Flagstaff", "held: opt_in anticon pool_held, rate 0.05, bind 127.0.0.1",
        ok(OPT_IN and ANTICON and POOL_HELD and RATE == 0.05 and BIND == "127.0.0.1"))
    add("Flagstaff", "second_bot mint socket layer_one fetch ue5_launch model_pull false",
        ok(not (SECOND_BOT or MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH or MODEL_PULL)))

    # FieldCore
    add("FieldCore", f"{len(texts)} text files decoded, undecodable {bad or 0}", PASS if not bad else ADV)
    wfs = sorted(r for r in texts if r.startswith(".github/workflows/") and r.endswith((".yml", ".yaml")))
    if not os.path.isdir(os.path.join(root, ".github")):
        add("FieldCore", ".github absent in this clone; workflow token/actor surface not checkable here", ADV)
    else:
        sec_names = sorted({f"{r}:{m.group(1)}" for r in wfs for m in WF_SECRET.finditer(texts[r])
                            if m.group(1) != "GITHUB_TOKEN"})
        if not sec_names:
            add("FieldCore", f"{len(wfs)} workflows use only secrets.GITHUB_TOKEN", PASS)
        for s in sec_names:
            add("FieldCore", f"workflow secret {s} (a non-default token; owner not readable)", ADV)
    add("FieldCore", f'PERSONAL_BOT_TECH keeps "{PERSONAL_ROW}"',
        ok(PERSONAL_ROW in texts.get("docs/PERSONAL_BOT_TECH.md", "")))
    return rows


def report(root: str, quiet: bool = False) -> tuple[int, int, int, str]:
    rows = checks(root)
    pk = pack([v for *_, v in rows])
    tip = GENESIS
    for side in SIDES:
        mine = [r for r in rows if r[0] == side]
        text = f"{ITEM_HEADING}|{SUBCHECK}|{side}|" + ";".join(f"{n}={v}" for _, n, v in mine)
        tip = close(tip, pk, text)
        if not quiet:
            for _, name, v in mine:
                print(f"{ {PASS: 'PASS', ADV: 'ADVISORY', FAIL: 'FAIL'}[v] } [{side}] {name}")
    p = sum(v == PASS for *_, v in rows)
    a = sum(v == ADV for *_, v in rows)
    f = sum(v == FAIL for *_, v in rows)
    return p, a, f, tip


def run() -> int:
    p, a, f, tip = report(ROOT)
    print(f'{ITEM_HEADING}: "{ITEM}" [{SUBCHECK}] pass {p} advisory {a} fail {f} '
          f"close {tip} rate {RATE} bind {BIND}")
    return 1 if f else 0


def _fixture(d: str, **over) -> None:
    files = {
        "docs/ROADMAP.md": f"# r\n## Order\n1. x\n\n## {ITEM_HEADING}\n{ITEM}\n",
        "docs/ZERO_TRUST.md": f"## Fail\n- {ZT_FAIL_ROW}\n\n## Identity\n"
                              "GitHub connector = cloudcover95. Automations share that login. "
                              "Not a second bot user.\n",
        "docs/TECH_AUDIT.md": f"## Do not ship\n- {TA_FAIL_ROW}\n",
        "docs/PERSONAL_BOT_TECH.md": f"Supplemental. {PERSONAL_ROW}. Owner: cloudcover95.\n",
        "docs/ledger/2026-10-20_0205.md": "All builder commits are authored as cloudcover95, "
                                          f"`cloudcover95 <{OWNER_EMAIL}>`.\n",
        "catalog/repos.json": json.dumps({"owner": OWNER, "repos": [
            {"name": "JuniorLLM", "role": "engine"}, {"name": OWNER, "role": "meta"}]}),
        "README.md": "[LLM](https://github.com/cloudcover95/JuniorLLM) "
                     "[up](https://github.com/SYSTRAN/faster-whisper)\n",
        "scripts/ok.sh": f'git config user.name {OWNER}\ngit config user.email {OWNER_EMAIL}\n',
        "deployment/docker-compose.yml": "services:\n  a:\n    image: x\n",
        ".git/config": '[remote "origin"]\n\turl = https://github.com/cloudcover95/JuniorHome\n'
                       f"[user]\n\tname = {OWNER}\n\temail = {OWNER_EMAIL}\n",
    }
    files.update(over)
    for rel, body in files.items():
        if body is None:
            continue
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)


def selftest() -> int:
    assert pack([1, -1, 0, 1]) == bytes([0b01100001])
    assert pack([1, 1, 1, 1, -1]) == bytes([0b01010101, 0b10000000])
    c1 = close(GENESIS, pack([1]), "a")
    det = c1 == close(GENESIS, pack([1]), "a") and c1 != close(GENESIS, pack([-1]), "a")
    tok = "gh" + "p_" + "A" * 36
    cases = {
        "clean": ({}, lambda p, a, f: f == 0 and a == 1 and p > 0),
        "second_bot_git_config": ({"scripts/bot.sh": "git config user.email junior-bot@example.com\n"},
                                  lambda p, a, f: f == 1),
        "deploy_identity_rewrite": ({"deployment/run.sh": f"git config user.email {OWNER_EMAIL}\n"},
                                    lambda p, a, f: f == 1),
        "env_author_py": ({"web3node/push.py": "env = {'GIT_AUTHOR_NAME': 'juniorbot'}\n"},
                          lambda p, a, f: f == 1),
        "token_literal": ({"web3node/cfg.py": f"T = '{tok}'\n"}, lambda p, a, f: f == 1),
        "token_env_adv": ({"web3node/cfg.py": "import os\nT = os.environ.get('GH_TOKEN')\n"},
                          lambda p, a, f: f == 0 and a == 2),
        "ledger_other_author": ({"docs/ledger/2026-10-21_0205.md": "authored as juniorbot.\n"},
                                lambda p, a, f: f == 1),
        "noreply_other": ({"docs/x.md": "by `bot <123+junior-bot@users.noreply.github.com>`\n"},
                          lambda p, a, f: f == 1),
        "catalog_repo_elsewhere": ({"docs/y.md": "https://github.com/juniorbot/JuniorLLM\n"},
                                   lambda p, a, f: f == 1),
        "identity_line_dropped": ({"docs/ZERO_TRUST.md": f"## Fail\n- {ZT_FAIL_ROW}\n"},
                                  lambda p, a, f: f == 1),
        "repo_user_other": ({".git/config": '[remote "origin"]\n\turl = https://github.com/cloudcover95/JuniorHome\n'
                                            "[user]\n\tname = juniorbot\n\temail = bot@example.com\n"},
                            lambda p, a, f: f == 2),
        "owner_decl_other": ({"docs/z.md": "Owner: juniorbot. Additive only.\n"}, lambda p, a, f: f == 1),
        "role_label_adv": ({"docs/ledger/2026-10-22_0205.md": "landed later, authored by the LLC account.\n"},
                           lambda p, a, f: f == 0 and a == 2),
        "roadmap_item_dropped": ({"docs/ROADMAP.md": "# r\n## Order\n1. x\n"}, lambda p, a, f: f == 2),
        "workflow_pat_adv": ({".github/workflows/ci.yml": "env:\n  T: ${{ secrets.BOT_PAT }}\n"},
                             lambda p, a, f: f == 0 and a == 1),
    }
    res = {}
    for name, (over, want) in cases.items():
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **over)
            p, a, f, tip = report(d, quiet=True)
            res[name] = bool(want(p, a, f)) and len(tip) == 16
            print(f"selftest {name}: pass {p} advisory {a} fail {f} close {tip} -> {'ok' if res[name] else 'BAD'}")
    good = det and all(res.values())
    print(f"selftest {'pass' if good else 'fail'}: {sum(res.values())}/{len(res)} cases, close_det={det}")
    return 0 if good and run() == 0 else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else run())
