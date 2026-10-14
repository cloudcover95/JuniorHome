# Grok bot tech sheet

Prelim bounds. JuniorCloud LLC. cloudcover95. Additive only.
This is the coordinator layer that charges the repos. It is not the local compute. Local compute is OSai + on the box the end user runs.

## What this is
Seven long-lived agents. One shared computer. They cannot see each other. Each owns a lane. The chief reads. The builders write. Chat is the control surface.

| Bot | Time MT | Owns |
|---|---|---|
| Chief of Staff | 02:00 | reads only. Audits tips. Flags stale. Does not commit. |
| JuniorHome Builder | 02:15 | JuniorHome, JuniorDrive, JuniorFetch |
| JuniorOS Builder | 02:45 | JuniorLLM, JuniorPiThon, JuniorPiPython |
| Engine Math | 03:15 | web3node, JuniorQuant, stocksnode, JuniorStock, JuniorSOL |
| Suite Scripts | 03:45 | JuniorPython-Suite, JuniorEngrTools, JuniorHome scripts |
| Spatial Suite | 04:15 | FrameForge, FrameForge2D, JuniorOmega, JuniorClimbs, JuniorCoach, JuniorPoker |
| Inference Core | 04:45 | BitNet-mlx, JuniorMemSys-Suite, crispy-mouse, JuniorAGI_SDK, AGI_SDK |

JuniorSolana is an archive. The profile repo is not a build target.

## Rules
- One repo per turn. One pull. One new file. One push.
- Additive only. Never delete. Never force-push. No `.github`.
- Bind 127.0.0.1. No 0.0.0.0.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Rate 0.05. Trit side-commit is sha256(prev || pack || text)[:16].
- No model pull. No GGUF download. Header read only if the file is already local.
- No 12-hour skip. If a turn drops, stop. Next run pulls.
- The chief adapts from the git tip, not from another bot's chat.

## Bounds toward local compute
These agents coordinate. They do not run the end-user stack.

| Bound | Here (bot) | There (OSai+) |
|---|---|---|
| Scope | one repo, one file | one note, one pack, one tick |
| Memory | transcript + git tip | local mesh, jsonl, no remote |
| Network | GitHub only | 127.0.0.1 only |
| Model | none pulled | local GGUF header or trit pack |
| Audit | 02:00 tip check | Flagstaff vote on the handshake |
| Fail | stop, next run pulls | raise the note, do not invent |

OSai+ owns the local path: handshake, Winsor, trit pack, I2_S, mesh append. The bot layer is the charge that keeps the code that does that current.

## What stays out
- No GitHub Actions, Copilot, Dependabot.
- No cloud model call from the suite.
- No bot that writes into another lane.
- No second chief. No second Home builder.

## Status
Prelim. The roster and lanes are the source of truth. This sheet does not override them. Update by adding a line, not by rewriting the table.
