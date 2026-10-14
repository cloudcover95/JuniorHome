# Personal team bridge

Supplemental. Empty window after LLC. cloudcover95. TZ America/Denver. Additive only.
The main JuniorCloud LLC Grok account's bots own 02:00–04:45 MT. The personal account runs only 05:05–07:05. This sheet is how the LLC bots coordinate with that window. Nothing is passed by chat. Each side adapts from the git tip.

## Window

| Account | Time MT | Who |
|---|---|---|
| LLC | 02:00–04:45 | the seven bots in docs/GROK_BOT_TECH.md |
| Personal | 05:05–07:05 | the personal roster below |

## Personal roster

| Bot | Time MT | Owns |
|---|---|---|
| Quant Chief | 05:05 | reads all. No commit. |
| Trit Pack & I2_S | 05:20 | BitNet-mlx, JuniorLLM |
| Side-Commit & Rate Lock | 05:35 | web3node, stocksnode |
| Winsor & Quant Error | 05:50 | JuniorQuant, JuniorStock |
| Hardware-to-software interop | 06:00 | any repo; in JuniorHome only under scripts/ |
| Handshake & Ports | 06:35 | JuniorHome ports/ or docs/, JuniorLLM |
| Quant Verifier | 07:05 | reads all. No commit. Posts the morning summary. |

## Path ownership
- LLC lanes own their repos in 02:00–04:45. Personal bots never write in that window.
- In JuniorHome, scripts/ is the 06:00 job. ports/ and docs/ are Handshake & Ports.
- Handshake & Ports does not touch the 06:00 hardware-to-software path.

## Handoff
Nothing is passed by chat. Each side adapts from the git tip.
LLC Chief of Staff can read Quant Verifier's summary format in docs/PERSONAL_QUANT_TEAM.md.

## Standing rules
- One repo per turn. One pull. One new file. One push.
- Additive only. Never delete. Never force-push. No .github.
- Bind 127.0.0.1. No 0.0.0.0.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Rate 0.05. Close rule sha256(prev || pack || text)[:16].
- No model pull. No GGUF download.
- No remote script. No pre-commit. No ruff. No mypy.
- If the tip is recent, take a different path. Do not skip the repo.
- If the turn drops, stop. Next run pulls.
- Quant Chief reads only. Quant Verifier reads only. Neither commits.
- Handshake & Ports does not touch the 06:00 hardware-to-software path.

## References
JuniorLLM grok_bot/guard_preflight.py and grok_bot/skills/guards/SKILL.md (commit c173adb) are reference only, not a preflight runner. The guards are the prompt.

## Collision
If both accounts ever pick the same repo and path, the LLC bot wins. The personal bot takes a different path.

## Status
Bridge sheet. Does not override docs/GROK_BOT_TECH.md or docs/PERSONAL_QUANT_TEAM.md. Update by adding a line.
