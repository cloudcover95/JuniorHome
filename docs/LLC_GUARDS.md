# LLC bot guards

Non-intrusive. Additive only. cloudcover95. JuniorCloud LLC.
These are the safety nets for the LLC bot roster. They do not slow a charge. They do not skip a recent tip. They check after the file is written.

## Roster

| Bot | Time MT | Repos |
|---|---|---|
| Chief of Staff | 02:00 | reads only |
| JuniorHome Builder | 02:15 | JuniorHome, JuniorDrive, JuniorFetch |
| JuniorOS Builder | 02:45 | JuniorLLM, JuniorPiThon, JuniorPiPython |
| Engine Math | 03:15 | web3node, JuniorQuant, stocksnode, JuniorStock, JuniorSOL |
| Suite Scripts | 03:45 | JuniorPython-Suite, JuniorEngrTools, JuniorHome scripts |
| Spatial Suite | 04:15 | FrameForge, FrameForge2D, JuniorOmega, JuniorClimbs, JuniorCoach, JuniorPoker |
| Inference Core | 04:45 | BitNet-mlx, JuniorMemSys-Suite, crispy-mouse, JuniorAGI_SDK, AGI_SDK |

The personal quant team runs 05:05–07:05. The LLC bots finish before it starts. No overlap. No 6-hour skip. No 12-hour skip.

## Guards (every builder)
1. One repo per turn. One pull. One new file. One push.
2. Additive only. Never delete. Never force-push. No `.github`.
3. Bind 127.0.0.1. No 0.0.0.0.
4. mint false. socket false. layer_one false. fetch false. ue5_launch false.
5. Rate 0.05 if the repo has a trit pack. Close rule sha256(prev || pack || text)[:16].
6. Self-check the new file. Name the ports it touches if it is a quant file: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
7. Confirm the engines still agree on the rate and the close rule.
8. No model pull. No GGUF download.
9. If the tip is recent, take a different path. Do not skip the repo.
10. If the turn drops, stop. Next run pulls.

## Chief guards
The chief reads. It does not build. It does not commit.
1. Read the tips of the roster repos.
2. Flag any repo with no additive commit in 48 hours. Name the lane and the repo.
3. Flag any commit to JuniorSolana or the cloudcover95 profile repo.
4. Flag any non-additive push.
5. Confirm the engines still agree on rate 0.05 and the side-commit rule.
6. Write one line per flag. No flag means ok.

## Automations
The in-house automations are guards, not builds. They report. They do not commit product code.

| Automation | Time MT | Job |
|---|---|---|
| overnight | 01:00 | Pre-window. Confirm the bots are ready. Do not skip. |
| daily audit | 16:45 | Post-window. List the tips. Confirm no GitHub Actions. |
| 19:00 | 19:00 | Evening. Confirm the personal team ran. Do not skip. |
| weekly audit | Mon 17:00 | Write WEEKLY_AUDIT.md. One commit. |
| omega-obj-weekly | Sat 10:00 | Confirm Omega files. Additive only if missing. |
| junioros-core-weekly | Sat 11:00 | Confirm OS ports. Additive only if missing. |

## What stays out
- No GitHub Actions, Copilot, Dependabot.
- No cloud model call from the suite.
- No bot that writes into another lane.
- No skip because a tip is recent.
- No product commit from an automation. The bots own the slices.

## Status
Live. This sheet does not override docs/ROSTER.md or docs/GROK_BOT_TECH.md. Update by adding a line.
