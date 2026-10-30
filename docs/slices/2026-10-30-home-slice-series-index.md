# Home slice 2026-10-30: slice series index

Read-only index of the JuniorHome Builder slices under `docs/slices/`, in date
order, with the close each one recorded. Docs only. No slice is edited,
nothing executed. Lane: JuniorHome Builder, docs. JuniorCloud LLC, cloudcover95.

## Slices

| Date | Path | What it covers | Close |
|---|---|---|---|
| 2026-10-15 | `docs/slices/2026-10-15-home-slice.md` | First run record: tip, path choice, trit pack, flags, guard sheets it points at | `28c1d4cf9359f556` |
| 2026-10-18 | `docs/slices/2026-10-18-os-state-map.md` | Which modules write which local state paths under `~/.juniorhome/` | `46b4dadfac4441bc` |
| 2026-10-21 | `docs/slices/2026-10-21-dash-tile-map.md` | The `ports/*_tile.py` dash tiles and the state file each one reads | `950d873d40612dd8` |
| 2026-10-24 | `docs/slices/2026-10-24-ports-handshake-index.md` | Handshake, port-map, ledger and gate modules in `ports/`, grouped by function shape | `5c1bb8af4e570c8e` |
| 2026-10-27 | `docs/slices/2026-10-27-guard-docs-crosscheck.md` | Cross-check of the five guard sheets and where they differ | `5838b019fd9bcd3f` |
| 2026-10-30 | `docs/slices/2026-10-30-home-slice-series-index.md` | This index | below |

## How they pair
- 2026-10-18 lists the writers of local state; 2026-10-21 lists the tiles that
  read it.
- 2026-10-21 covers the `*_tile.py` readers in `ports/`; 2026-10-24 covers the
  other `ports/` modules.
- 2026-10-15 lists the guard sheets; 2026-10-27 cross-checks them.

## Close inputs used so far
- 2026-10-15 recorded pack as `packs/junior_trit_stack/pack.json` (by file hash).
- 2026-10-27 recorded pack as the pack id `junior.trit_stack`.
- This slice uses the pack id `junior.trit_stack`, as 2026-10-27 did.

## Cadence
JuniorHome takes every third run in the home-slice rotation (JuniorHome, then
JuniorDrive, then JuniorFetch), so slices land here three days apart.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Not a quant file; no port code read or changed. No model pull, no remote
  script, no repo code run. No `web3node/` or `scripts/` touch.
- Tip was recent (`7b09619`, 2026-10-30 02:16 MT), so this takes a new slice path.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `7b09619976d8417803bb8d9282d61f4f7117352a`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `805e9183bc88c82c` · rate 0.05
