# JuniorHome / JuniorOS roadmap after 2026-09-13 cap

Slots **07:00 and 13:00 paused**. Live schedule:
- 01:00 overnight — owns T-slices (next T13 height)
- 19:00 — second slice if receipt >45 min old
- 16:45 daily audit
- Sun 17:00 weekly (dd0b1c93)
- Fri 16:45 weekly still on (7716afb1) — consider pause later; duplicate burn

## Order (do not skip)
1. **T13+** — juniorctl skill-pin height, then T14 only after receipt. JuniorLLM rails/linux. Overnight owns this.
2. **GGUF** — operator copies a local file; set JUNIOR_GGUF. Bots do not download. llama_ready flips when path exists. Probe: `web3node/probe_future.py`.
3. **i2sd userspace** — already sketched in JuniorLLM rails/linux. Vendor kernel daemon is opt-in on the box; loopback 127.0.0.1:8767. No docker.sock.
4. **Asahi MLX** — probe mlx + asahi sysfs if present; no kernel patch from chat.
5. **UE5 launched surface** — os_route.launch stays false until watts>=80 AND JUNIOR_UE5=1 on the box. Default is FrameForge2D.

## Not this roadmap
0.0.0.0 SaaS plane, 4B pulls, MP/KAYA scrape, new GitHub bot user.
