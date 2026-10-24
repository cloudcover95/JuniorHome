# Home slice 2026-10-24: ports handshake index

Read-only index of the handshake, port-map, ledger and gate modules in
`ports/`, grouped by the shape of their functions. Pairs with
`docs/slices/2026-10-21-dash-tile-map.md` (the `*_tile.py` readers).
Docs only. No code change, nothing executed. Lane: JuniorHome Builder, docs.
JuniorCloud LLC, cloudcover95.

Ports named across these files: Home, OSai, FieldCore, Flagstaff, JuniorLLM.

## Shape A: close chain (`pack`, `close`, `anticon`, `probe`, `port_map`, `build_chain`, `verify_chain`, `handshake_check`, `self_check`)

| File | Docstring subject |
|---|---|
| `ports/asahi_mlx_probe_handshake.py` | ROADMAP #4 Asahi MLX, probe only from automations |
| `ports/bitnetd_loopback_handshake.py` | bitnetd loopback, ROADMAP named item |
| `ports/daily_audit_handshake.py` | 16:45 daily audit |
| `ports/gguf_operator_handshake.py` | ROADMAP #2 GGUF, operator sets `JUNIOR_GGUF`, bots do not download |
| `ports/i2sd_userspace_handshake.py` | ROADMAP #3 i2sd userspace, 127.0.0.1:8767, no docker.sock |
| `ports/i2sd_userspace_8767_handshake.py` | i2sd userspace 8767, ROADMAP named item |
| `ports/i2sd_userspace_8767_named.py` | same subject as the line above |
| `ports/juniorctl_health_handshake.py` | BETA_TO_OS live beta item 1 |
| `ports/juniorctl_ask_night_security_handshake.py` | BETA_TO_OS live beta item 2 |
| `ports/juniorctl_skill_pin_t13_handshake.py` | juniorctl skill-pin height, T13+ |
| `ports/t13_skill_pin_handshake.py` | ROADMAP #1 T13+, overnight owns this |
| `ports/overnight_t13_height_handshake.py` | same subject as the line above |
| `ports/overnight_t_slices_handshake.py` | 01:00 overnight T-slices |
| `ports/junioros_overlay_handshake.py` | JuniorOS overlay, ROADMAP named item |
| `ports/second_slice_receipt_handshake.py` | second-slice receipt, ROADMAP named item |
| `ports/ue5_launched_handshake.py` | ROADMAP #5 UE5 launched, `JUNIOR_UE5=1` and watts>=80 |
| `ports/ue5_frameforge2d_default_handshake.py` | same subject as the line above, default FrameForge2D |

## Shape B: side-commit chain (`pack`, `side_commit`, `port_map`, `build_chain`, `verify_chain`, `handshake_check`, `self_check`)

| File | Docstring subject |
|---|---|
| `ports/handshake_port_map.py` | handshake check and port map across the five ports |
| `ports/daily_handshake_check.py` | daily handshake check across the five ports |
| `ports/gguf_bitnetd_handshake.py` | GGUF to bitnetd, bots do not download |
| `ports/ue5_launch_gate.py` | UE5 launch gate, `JUNIOR_UE5=1` and watts>=80 |
| `ports/overlay_base_handshake.py` | overlay base, Debian/Alpine split, juniorctl on PATH, loopback ports |

## Shape C: ledgers and registries (`close`, `handshake` or `run`, `selftest`, `main`)

| File | Docstring subject |
|---|---|
| `ports/gguf_download_deny_ledger.py` | GGUF download-deny provenance ledger |
| `ports/i2sd_userspace_privilege_ledger.py` | i2sd privilege and loopback port ledger |
| `ports/i2sd_userspace_doc_port_registry.py` | who owns 127.0.0.1:8767 across Home docs |
| `ports/bitnetd_i2sd_handover_handshake.py` | bitnetd to i2sd handover, BETA_TO_OS overlay item |

## Shape D: doc port maps (parse docs, `evaluate` or `run`, `selftest`)

| File | Docstring subject |
|---|---|
| `ports/daily_audit_slot_port_map.py` | ROADMAP Automations vs `AUTOMATION_AUDIT.md` |
| `ports/daily_audit_remaining_engines_port_map.py` | BitNet-mlx, web3node, JuniorQuant |
| `ports/chief_weekly_receipt_port_map.py` | ROADMAP chief weekly vs its receipts (`WEEKLY_AUDIT.md`) |
| `ports/deck_sheet_port_map.py` | JuniorDeck pin table to package ports |
| `ports/overlay_netinst_path_port_map.py` | juniorctl on PATH per Debian/Alpine install lane |

## Notes
- Every `0.0.0.0` in these files sits in a negative self-test fixture or a
  "never" line; none is a bind.
- Three pairs share a docstring subject but are not byte-identical:
  the two i2sd 8767 files, the two T13 files, and the two UE5 files.
  `handshake_port_map.py` and `daily_handshake_check.py` also differ.
  Noted only; `ports/` code is not edited by this slice.
- Not indexed here: `*_tile.py` (see the 2026-10-21 slice), `pointers.py`,
  `lab_*.py`, and `blender_omega.py`.

## Guards
- One new file. Additive only. No `.github`. Bind 127.0.0.1.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Index of quant-adjacent handshakes, so the five ports are named above.
  No model pull, no remote script, no repo code run.
- Tip was recent (2026-10-24 02:11 MT), so this takes a new slice path.
- Repo has a trit pack (`packs/junior_trit_stack`, id `junior.trit_stack`), so rate 0.05.
- Close rule sha256(prev || pack || text)[:16], with prev = tip `adb401f9938e50b33db6d33e13378eb39920588c`,
  pack = `junior.trit_stack`, text = this file above the close line.

close: `5c1bb8af4e570c8e` · rate 0.05
