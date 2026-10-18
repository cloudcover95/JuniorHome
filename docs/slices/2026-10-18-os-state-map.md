# Home slice 2026-10-18: local state file map

Read-only map of which JuniorHome modules write which local state paths
under `~/.juniorhome/`. Docs only. No code change, nothing executed.
Lane: JuniorHome Builder, docs. JuniorCloud LLC, cloudcover95.

| Module | Writes |
|---|---|
| `web3node/deps.py` | `~/.juniorhome/os/deps.json` |
| `web3node/native_home.py` | `~/.juniorhome/os/native_home.json` |
| `web3node/safety_net.py` | `~/.juniorhome/os/safety_net.json` |
| `web3node/prod_manager.py` | `~/.juniorhome/os/prod_manager.json` |
| `web3node/os_mesh.py` | `~/.juniorhome/gaia_mesh/os_mesh.jsonl` |
| `web3node/surface_spin.py` | `~/.juniorhome/surfaces/<kind>/ticket.json` (kind: react-native, web, legacy) |
| `web3node/team_report.py` | `~/.juniorhome/obsidian/findings/<YYYY-MM-DD_HHMM UTC>_<team>.md` and appends `~/.juniorhome/brain/team.jsonl` |

All of these are local files. `safety_net`, `os_mesh`, `surface_spin`, and
`team_report` carry `bind: 127.0.0.1`. `surface_spin` tickets keep
`metro: false` and `live: false`.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `efb071dd4342dc244e267d5be1099a24fac53c7d`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `46b4dadfac4441bc` · rate 0.05
