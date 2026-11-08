# Home slice 2026-11-08: slice series index (continued)

Read-only continuation of `docs/slices/2026-10-30-home-slice-series-index.md`.
Lists the JuniorHome Builder slices under `docs/slices/` added after that
index, in date order, with the close each one recorded. Docs only. No slice
is edited, nothing executed. Lane: JuniorHome Builder, docs. JuniorCloud LLC,
cloudcover95.

## Prior index

`docs/slices/2026-10-30-home-slice-series-index.md` covers 2026-10-15 through
2026-10-30 (close `805e9183bc88c82c`). This file does not replace it.

## Slices after 2026-10-30

| Date | Path | What it covers | Close |
|---|---|---|---|
| 2026-11-02 | `docs/slices/2026-11-02-deck-sheet-index.md` | The `docs/DECK_*.md` JuniorDeck sheets, with first line and line count | `0e3094da6c9d868d` |
| 2026-11-05 | `docs/slices/2026-11-05-tuner-sheet-index.md` | The `docs/TUNER*.md` Tuner sheets, with first line and line count | `51478fc863b9b54f` |
| 2026-11-08 | `docs/slices/2026-11-08-home-slice-series-index-continued.md` | This continuation | below |

## How they pair
- 2026-11-02 indexes the deck sheets the same way 2026-11-05 indexes the
  tuner sheets.
- 2026-10-30 listed the series through that date; this note lists the
  slices that landed after it.

## Close inputs used so far
- 2026-10-30 and the slices after it use the pack id `junior.trit_stack`.
- This slice uses the pack id `junior.trit_stack`, as 2026-10-30 and
  2026-11-05 did.

## Cadence
JuniorHome takes every third run in the home-slice rotation (JuniorHome, then
JuniorDrive, then JuniorFetch), so slices land here three days apart.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Not a quant file; no port code read or changed. No model pull, no remote
  script, no repo code run. No `web3node/` or `scripts/` touch.
- Tip was recent (`c69ef33`, 2026-11-08 02:16 MT), so this takes a new slice path.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `c69ef33872bd605042bde7e04ea24ca734094a19`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `675b8d0ed1d86418` · rate 0.05
