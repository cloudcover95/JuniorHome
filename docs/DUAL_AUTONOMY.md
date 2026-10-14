# Dual autonomy

Two Grok accounts. One owner. They do not share a transcript. They share the repos.
cloudcover95. Additive only. JuniorCloud LLC.

## Windows

| Account | Window MT | Owns |
|---|---|---|
| LLC | 02:00–04:45 | Home, OS, math, scripts, spatial, inference |
| Personal | 05:05–07:05 | quant only: trit, I2_S, side-commit, Winsor, handshake |
| Automations | 01:00, 16:45, 19:00, weekly | report only. No product commit. |

The 06:00 hardware-to-software job on the personal account writes to JuniorHome. Handshake & Ports must not touch that path.

## No remote code
No bot downloads and executes a file from main before a push. The guards live in the prompt. A script on the repo is a file, not a preflight runner. `grok_bot/guard_preflight.py` on JuniorLLM is a reference. It is not fetched and run.

## Shared rules
- One repo per turn. One pull. One new file. One push.
- Additive only. Never delete. Never force-push. No `.github`.
- Bind 127.0.0.1. No 0.0.0.0.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Rate 0.05. Close rule sha256(prev || pack || text)[:16].
- No model pull. No GGUF download.
- No 6-hour skip. No 12-hour skip. If the tip is recent, take a different path.
- If the turn drops, stop. Next run pulls.
- JuniorSolana is an archive. cloudcover95 is the profile repo. Do not build there.

## What each side does not do
- The LLC bots do not write into the personal team window.
- The personal team does not write into the LLC window.
- Neither side can see the other. Coordination is the tip on GitHub.
- Neither side fetches remote code to run.

## Status
Live. This sheet does not override docs/ROSTER.md, docs/LLC_GUARDS.md, or docs/PERSONAL_QUANT_TEAM.md. Update by adding a line.
