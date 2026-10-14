# Suite glue: JuniorHome ↔ JuniorDrive ↔ JuniorFetch

How the three JuniorHome Builder repos (see `docs/ROSTER.md`) touch each other, read from the repos at
JuniorHome f9d2fb1, JuniorDrive 1620f37, JuniorFetch 5ea9cc3. Notes only. Nothing here opens a socket or fires.
All binds are 127.0.0.1 only. Never 0.0.0.0.

## The shared seam: `~/.juniorhome/os/lean.json`

Both siblings ship an identical `lean_read.py` that reads a file under Home's state dir:

| Repo | File | If `lean.json` is missing |
|---|---|---|
| JuniorDrive | `lean_read.py` | `{"ok": false, "port": "JuniorDrive", "writes": 0}` |
| JuniorFetch | `lean_read.py` | `{"ok": false, "port": "JuniorFetch", "writes": 0}` |

Read only. Neither sibling writes into `~/.juniorhome/`. This is the only code-level link found;
no sibling imports a JuniorHome module, and `ports/pointers.py` lists neither sibling.

## Shared local endpoints (from each sibling's `docs/LOCAL.md`)

| Service | Bind | Home doc |
|---|---|---|
| hook | 127.0.0.1:8770 | `docs/HOOK.md` |
| i2sd | 127.0.0.1:8767 | `docs/OBSIDIAN.md`, `docs/INTEGRATION.md` |

Both LOCAL.md files also name Obsidian at `JuniorHome/vault/bitnetCloud` (not present at this tip)
and quant/TP from JuniorLLM `junior_bitnet` (AbsMean 1.58, column-shard).

## What each sibling brings

- **JuniorDrive** — `core/router.py` `route()`: profiles `T4_ship` (default, ternary-1.58, ticket),
  `T0_m6` (ternary-1.58, kernel+local-gguf), `T1_spark` (fine-tune latch, not 70B). Unknown names fall back to `T4_ship`.
- **JuniorFetch** — `juniorfetch` CLI (`index`, `dashboard`). `JuniorFetchCrawler` indexes text-like files into
  `MemoryPalace` at `~/.juniorfetch/palace` (env prefix `JUNIORFETCH_`). Its own state dir, separate from `~/.juniorhome/`.
  `core/llm_sandbox.py` loads a model on construction; do not construct it in automated runs.

## Rules carried across

Additive only. 127.0.0.1 only. No GitHub Actions. No model pulls.
See also: `docs/PORTS_INDEX.md`, `docs/ECOSYSTEM_PORTS.md`, `docs/LANES.md`, `docs/ROSTER.md`.
