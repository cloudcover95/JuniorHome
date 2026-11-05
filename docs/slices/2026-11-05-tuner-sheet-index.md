# Home slice 2026-11-05: tuner sheet index

Read-only index of the Tuner sheets in `docs/` (`docs/TUNER*.md`), with each
sheet's first line and line count as they stand at tip `dce0b70`. Docs only.
No sheet is edited, nothing executed. Lane: JuniorHome Builder, docs.
JuniorCloud LLC, cloudcover95.

## Sheets

| Sheet | First line | Lines |
|---|---|---|
| `docs/TUNER.md` | Tuner | 28 |
| `docs/TUNER_CLI.md` | Tuner CLI | 16 |
| `docs/TUNER_HARDWARE.md` | Tuner hardware | 20 |
| `docs/TUNER_LEDGER.md` | Tuner ledger | 14 |
| `docs/TUNER_MANAGER.md` | Tuner manager | 22 |
| `docs/TUNER_MODULE.md` | Tuner module | 13 |
| `docs/TUNER_PROFILES.md` | Tuner hardware profiles | 18 |
| `docs/TUNER_STACK.md` | Tuner stack hook | 10 |

All eight sheets open with a `#` heading; the heading text is listed above.

## Related files outside docs/ (names only, not read or changed)
- `ports/tuner_stack_tile.py`

## Where it sits in the series
Follows `docs/slices/2026-11-02-deck-sheet-index.md` (close
`0e3094da6c9d868d`), which indexed the deck sheets the same way. This one
indexes the tuner sheets.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Not a quant file; no port code read or changed. No model pull, no remote
  script, no repo code run. No `web3node/` or `scripts/` touch.
- Tip was recent (`dce0b70`, 2026-11-05 02:14 MT), so this takes a new slice path.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `dce0b70cb6d6f1e7f5f67072aeb207024ca6e3b2`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `51478fc863b9b54f` · rate 0.05
