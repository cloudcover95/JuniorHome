# JuniorHome / JuniorOS roadmap after 2026-09-13 cap

## Automations (current)
ON:
- 01:00 overnight — T-slices (next T13 height)
- 19:00 — second slice if receipt >45 min old
- 16:45 daily audit
- **JuniorCloud chief weekly** Sun/Mon 17:00 MDT (`dd0b1c93`) — only weekly. Writes `docs/WEEKLY_AUDIT.md`.

PAUSED:
- 07:00, 13:00
- Friday weekly `7716afb1` (merged into chief)

Do not recreate paused slots.

## Order (do not skip)
1. **T13+** — juniorctl skill-pin height. Overnight owns this.
2. **GGUF** — operator sets JUNIOR_GGUF. Bots do not download. `web3node/probe_future.py`.
3. **i2sd userspace** — 127.0.0.1:8767. No docker.sock.
4. **Asahi MLX** — probe only from automations.
5. **UE5 launched** — `JUNIOR_UE5=1` and watts>=80. Default FrameForge2D. `os_route.launch` false until then.

## Not this roadmap
0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user.
