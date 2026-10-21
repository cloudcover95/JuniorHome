# Home slice 2026-10-21: dash tile map

Read-only map of the `ports/*_tile.py` dash tiles: which local state file
each tile reads, which module writes it, and which keys the tile surfaces.
Pairs with `docs/slices/2026-10-18-os-state-map.md` (writers). Docs only.
No code change, nothing executed. Lane: JuniorHome Builder, docs.
JuniorCloud LLC, cloudcover95.

| Tile | Reads | Written by | Keys surfaced |
|---|---|---|---|
| `ports/deps_tile.py` | `~/.juniorhome/os/deps.json` | `web3node/deps.py` | `modules`, `local`, plus computed `present`, `total`, `absent` |
| `ports/native_home_tile.py` | `~/.juniorhome/os/native_home.json` | `web3node/native_home.py` | `home_present`, `os_dir`, `native`, `local` |
| `ports/safety_net_tile.py` | `~/.juniorhome/os/safety_net.json` | `web3node/safety_net.py` | `ok`, `safety`, `rate`, `local` |
| `ports/prod_manager_tile.py` | `~/.juniorhome/os/prod_manager.json` | `web3node/prod_manager.py` | `team`, `balance_ok`, `profit_rows`, `kernel_viable`, `ternary_mem`, `stack`, `safety`, `local` |
| `ports/brain_digest_tile.py` | `~/.juniorhome/os/brain_digest.json` | `web3node/brain_digest.py` | `rows`, `n` |
| `ports/osai_plan_tile.py` | `~/.juniorhome/os/osai_plan.json` | `web3node/osai_hook.py` | `m`, `k`, `layer`, `bench`, `lut` |
| `ports/tuner_stack_tile.py` | `~/.juniorhome/os/tuner_stack.json` | `web3node/tuner/stack_hook.py` | `m`, `k`, `layer`, `bench`, `lut` |
| `ports/findings_tile.py` | latest `~/.juniorhome/obsidian/findings/*.md` by mtime | `web3node/team_report.py` | `path`, `name`, `mtime` (UTC), `text` (first 4000 chars), `truncated` |

## Shared shape
- Every tile is a single `tile(path)` function that only reads a local file.
  No network, no socket, no write.
- Each result carries `tile`, the path it read, and `status`: `ok`,
  `missing` (no file yet), or `unreadable` (read or JSON error, with `error`).
- On `missing` or `unreadable` the payload key is `None`, so a dash can
  render an empty tile instead of failing.
- Each tile prints its own result as JSON when run directly. This slice did
  not run any of them.

## Gaps seen
- `osai_plan_tile.py` does not check that the JSON is an object before
  calling `.get`, unlike the other JSON tiles. Noted only; `ports/` code is
  not edited by this slice.
- No tile yet for `os_mesh` (`~/.juniorhome/gaia_mesh/os_mesh.jsonl`) or the
  `surface_spin` tickets (`~/.juniorhome/surfaces/<kind>/ticket.json`).

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Not a quant file. No model pull, no remote script.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `640d959cfa9388b17473fffac2872cab092caa21`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `950d873d40612dd8` · rate 0.05
