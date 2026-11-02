# Home slice 2026-11-02: deck sheet index

Read-only index of the JuniorDeck sheets in `docs/` (`docs/DECK_*.md`), with
each sheet's first line and line count as they stand at tip `09aebf7`. Docs
only. No sheet is edited, nothing executed. Lane: JuniorHome Builder, docs.
JuniorCloud LLC, cloudcover95.

## Sheets

| Sheet | First line | Lines |
|---|---|---|
| `docs/DECK_AUDIO.md` | JuniorDeck audio → OSai | 13 |
| `docs/DECK_BOARD.md` | JuniorDeck board class | 15 |
| `docs/DECK_BOM.md` | JuniorDeck BOM | 9 |
| `docs/DECK_BUS.md` | Deck bus | 8 |
| `docs/DECK_CAPABILITY.md` | JuniorDeck capability | 8 |
| `docs/DECK_CHANNELS.md` | Deck channels | 10 |
| `docs/DECK_ENGINE.md` | JuniorDeck engine | 10 |
| `docs/DECK_FEED.md` | Channel map feed | 9 |
| `docs/DECK_HW.md` | JuniorDeck hardware | 6 |
| `docs/DECK_HW_PROD.md` | Production hardware admit | 5 |
| `docs/DECK_INPUT.md` | Deck input | 10 |
| `docs/DECK_LIVE.md` | python3 web3node/check_deck.py | 2 |
| `docs/DECK_MATH.md` | Deck math | 11 |
| `docs/DECK_MIDI.md` | MIDI | 9 |
| `docs/DECK_NATIVE.md` | Native deck hardware | 6 |
| `docs/DECK_ROADMAP.md` | JuniorDeck modular port | 12 |
| `docs/DECK_SESSION.md` | Deck session | 8 |
| `docs/DECK_SHELL.md` | JuniorDeck shells | 5 |
| `docs/DECK_TECH.md` | JuniorDeck tech sheet | 9 |
| `docs/DECK_USB.md` | JuniorDeck USB-C digitizer layer | 14 |

`docs/DECK_LIVE.md` has no heading; its first line is a command, listed as
text only. It was not run.

## Related files outside docs/ (names only, not read or changed)
- `ports/deck_sheet_port_map.py`
- `config/deck_audio.toml`, `config/deck_board.toml`, `config/deck_port.toml`
- `cad/deck_keys.scad`, `cad/deck_shell.scad`, `cad/deck_shell_field.scad`,
  `cad/deck_trit_band.scad`, `cad/deck_well.scad`

## Where it sits in the series
Follows `docs/slices/2026-10-30-home-slice-series-index.md` (close
`805e9183bc88c82c`). Earlier slices mapped state writers, dash tiles, ports
modules and guard sheets; this one maps the deck sheets.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Not a quant file; no port code read or changed. No model pull, no remote
  script, no repo code run. No `web3node/` or `scripts/` touch.
- Tip was recent (`09aebf7`, 2026-11-02 02:14 MT), so this takes a new slice path.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `09aebf7a84d3d6ff12c51b719fb7be0f31da57b0`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `0e3094da6c9d868d` · rate 0.05
