# Home slice 2026-10-27: guard docs cross-check

Read-only cross-check of the five guard sheets the LLC builders read before a
push: `docs/LLC_GUARDS.md`, `docs/DUAL_AUTONOMY.md`, `docs/BOT_FIRST.md`,
`docs/LANES.md`, `docs/ROSTER.md`. Lists which shared rule each sheet states and
where the sheets differ. Docs only. No sheet is edited, nothing executed.
Lane: JuniorHome Builder, docs. JuniorCloud LLC, cloudcover95.

## Sheets read (blob SHAs at tip `4bd6d2a`)

| Sheet | Blob | Lines |
|---|---|---|
| `docs/LLC_GUARDS.md` | `3978d2d3` | 61 |
| `docs/DUAL_AUTONOMY.md` | `3338a275` | 37 |
| `docs/BOT_FIRST.md` | `aa53690a` | 34 |
| `docs/LANES.md` | `94b6a239` | 16 |
| `docs/ROSTER.md` | `20edea3f` | 17 |

## Shared rules, by sheet

| Rule | LLC_GUARDS | DUAL_AUTONOMY | BOT_FIRST | LANES | ROSTER |
|---|---|---|---|---|---|
| One repo, one new file, one push | yes | yes | yes (one shallow clone) | yes (one slice) | yes |
| Additive only | yes | yes | yes | yes | yes |
| No GitHub Actions / `.github` | yes | yes (`.github`) | no | yes | yes |
| Bind 127.0.0.1 | yes | yes | no | yes | yes |
| mint, socket, layer_one false | yes | yes | no | yes | yes |
| fetch false, ue5_launch false | yes | yes | no | no | no |
| Rate 0.05 | yes (if trit pack) | yes | no | yes (Juniors) | no |
| Close rule sha256(prev \|\| pack \|\| text)[:16] | yes | yes | no | no | no |
| No model pull, no GGUF download | yes | yes | no | no | no |
| No remote script run | no | yes | yes (`guard_preflight.py`) | no | no |
| Recent tip: new path, no skip | yes | yes | yes | yes (no 12-hour skip) | yes (no 12-hour skip) |
| Turn drops: stop, next run pulls | yes | yes | no | yes | yes |
| JuniorSolana archive, profile repo off-limits | yes (chief flags) | yes | no | no | yes |

## Windows and slots

- LLC window 02:00–04:45 MT and personal quant window 05:05–07:05 MT are stated
  in `DUAL_AUTONOMY.md` and `LLC_GUARDS.md`.
- JuniorHome Builder is 02:15 MT in `LLC_GUARDS.md`, `LANES.md` and `ROSTER.md`.
- The automation slots (01:00, 16:45, 19:00, Mon 17:00, Sat 10:00, Sat 11:00) are
  listed only in `LLC_GUARDS.md`; `DUAL_AUTONOMY.md` lists them as report only.

## Differences noted (not edited)

- `LANES.md` says five bots and lists Chief, JuniorHome Builder, JuniorOS Builder,
  Engine Math and Suite Scripts. `ROSTER.md` and `LLC_GUARDS.md` list seven, adding
  Spatial Suite (04:15) and Inference Core (04:45).
- `LANES.md` gives JuniorHome Builder the JuniorHome repo only; `ROSTER.md` and
  `LLC_GUARDS.md` give it JuniorHome, JuniorDrive and JuniorFetch.
- `LANES.md` scopes Engine Math to JuniorHome `web3node` trit and juniors;
  `ROSTER.md` gives it the web3node, JuniorQuant, stocksnode, JuniorStock and
  JuniorSOL repos.
- `LANES.md` gives JuniorOS Builder JuniorLLM only; `ROSTER.md` adds JuniorPiThon
  and JuniorPiPython.
- `BOT_FIRST.md` step 3 is a one-line pre-push report to the chief; the other
  sheets do not mention it.
- `DUAL_AUTONOMY.md` notes that the 06:00 personal hardware-to-software job writes
  to JuniorHome and Handshake & Ports must not touch that path.

Each sheet says it does not override the others and is updated by adding a line,
so these are recorded here and left as they are.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Not a quant file; no port code read or changed. No model pull, no remote
  script, no repo code run.
- Tip was recent (`4bd6d2a`, 2026-10-27 02:13 MT), so this takes a new slice path.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `4bd6d2a6702e477d164f86c49aa312084b83312f`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `5838b019fd9bcd3f` · rate 0.05
