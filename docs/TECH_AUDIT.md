# JuniorHome tech audit 2026-09-13

Ignore community / nonprofit framing. Home is the box.

## Live compute
- JuniorLLM STATE: T12 juniorctl skill-pin log shipped. Next T13 height.
- Bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd. Not 0.0.0.0.
- llama_ready: false until JUNIOR_GGUF on disk.
- Trit pack: JuniorLLM junior_bitnet/winsor.py (stdlib {-1,0,1}).

## Do not ship from paste
- FastAPI host 0.0.0.0:8080
- docker-compose network_mode host
- git config identity override in deploy scripts
- 4B checkpoint downloads

## Repos updated last 48h
JuniorHome, JuniorLLM, JuniorDrive, JuniorOmega, JuniorEngrTools, FrameForge, BitNet-mlx, JuniorMemSys-Suite.
26 cloudcover95 repos. Compute stays in JuniorLLM + Home pointers.

## Efficiency (Grok quota vs API)
Automations spend SuperGrok weekly reset. xAI HTTP API is extra cash after promo.
Keep 01:00 + one audit. Pause extra daily slots if chat hits the cap.
