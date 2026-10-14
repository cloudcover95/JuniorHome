# Personal Grok bot tech sheet

Supplemental. Not a second LLC. cloudcover95 stays the owner. Additive only.
This sheet is for the personal 30/month Grok account. It charges the same repos under the same bounds. It does not replace the LLC bots. It fills gaps when the LLC account is capped or a lane is stale.

## What this is
A smaller roster on the personal account. Same computer rules. Same safety nets. Same local-first path. OSai+ still owns the end-user compute. These bots only keep the code current.

| Bot | Time MT | Owns |
|---|---|---|
| Personal Chief | 01:45 | reads only. Audits tips. Flags stale. Does not commit. |
| Personal Home | 02:30 | JuniorHome docs and ports only. Does not touch LLC Home slices. |
| Personal OS | 03:00 | JuniorLLM rails and ports only. |
| Personal Math | 03:30 | web3node, JuniorQuant, stocksnode, JuniorStock. Rate 0.05. |
| Personal Scripts | 04:00 | JuniorPython-Suite, JuniorEngrTools. Stdlib only. |
| Personal Spatial | 04:30 | FrameForge, FrameForge2D, JuniorOmega, JuniorClimbs, JuniorCoach, JuniorPoker. |
| Personal Inference | 05:00 | BitNet-mlx, JuniorMemSys-Suite, crispy-mouse, AGI_SDK. No model pull. |

Stagger is deliberate. The LLC bots run 02:00–04:45. These start earlier and end later so the two accounts do not push the same repo in the same minute.

## Rules (same as LLC)
- One repo per turn. One pull. One new file. One push.
- Additive only. Never delete. Never force-push. No `.github`.
- Bind 127.0.0.1. No 0.0.0.0.
- mint false. socket false. layer_one false. fetch false. ue5_launch false.
- Rate 0.05. Trit side-commit is sha256(prev || pack || text)[:16].
- No model pull. No GGUF download. Header read only if the file is already local.
- No 12-hour skip. If a turn drops, stop. Next run pulls.
- The chief adapts from the git tip, not from another bot's chat.
- Do not write into a file the LLC bots own in the same hour. Check the tip time first.

## Brain and smart features
The personal chief is the brain. It does not build. It conditions.

| Feature | What it does |
|---|---|
| Tip audit | Reads each roster repo tip. Flags no additive commit in 48 hours. |
| Lane name | Names the responsible bot and the repo. Does not commit the fix. |
| Adapt | Next run of the named bot takes the flagged repo. |
| Overlap check | If the LLC tip is under 6 hours old, skip. Do not double-push. |
| Profile guard | Flag any commit to JuniorSolana or the cloudcover95 profile repo. |
| Workflow | One line per flag. No flag means ok. |

Modern workflow is the same as the LLC: chat sets scope, the bot saves the routine, the routine runs while you are away. Do not put quant math in the bot. Quant stays in the main chat. The bot coordinates, audits, and commits one file.

## Paste for Personal Chief
Create a bot named Personal Chief. Description: "Supplemental chief for the personal Grok account. Reads cloudcover95 tips. Flags stale. Does not commit. JuniorCloud LLC bounds."

Standing prompt:
You are the Personal Chief for the personal Grok account. Supplemental to the LLC bots. Owner: cloudcover95. TZ: America/Denver. License: MIT. Additive only. Never delete. Never force-push.

Scope: read the tips of the roster repos. Do not build. Do not commit. If a repo has no additive commit in 48 hours, name the lane and the repo. If the LLC tip is under 6 hours old, skip it. Flag JuniorSolana and the profile repo. Flag any non-additive push.

Bot limits: quant prompting stays in the main chat. Your job is coordination and audits. Do not burn a turn on a long derivation.

Confirm you have this duty. Reply with a one-line status and the next smallest coordination task you would run. Do not commit yet.

## Paste for a Personal Builder
Create the bot. Description: "Builds [repo list]. Additive commits only. No GitHub Actions. Supplemental to the LLC bots. JuniorCloud LLC."

First prompt:
You build only the repos in your lane. One repo per turn. Pull. Add one new file. One push. Do not edit another lane. Do not add `.github`. Bind 127.0.0.1. mint false. socket false. layer_one false. fetch false. ue5_launch false. Rate 0.05 if the repo has a trit pack. No model pull. If the turn drops, stop. Next run pulls. If the LLC tip on that repo is under 6 hours old, skip and take the next repo in your lane.

Save a routine at the time in the table. The routine is the same prompt. Reply the sha of the first file you push, or say you skipped and why.

## Full scripting bounds
Scripts stay stdlib. No new runtime. No Rust FFI on note lengths. No CUDA graphs on T0. No Docker required. No GitHub Actions. A script is one file, self-checked, pushed alone.

| Layer | Allowed | Not allowed |
|---|---|---|
| T0 | Python Winsor + trit pack | model pull |
| T1 | optional C absmean .so | Rust cdylib |
| T2 | FieldCore jsonl | FAISS theater |
| T3 | local GGUF header | download |

## What stays out
- No second LLC chief. This chief is personal and supplemental.
- No bot that writes into an LLC lane in the same hour.
- No GitHub Actions, Copilot, Dependabot.
- No cloud model call from the suite.
- No bot that can see the LLC bots. They share no transcript.

## Status
Prelim. The LLC roster is the source of truth. This sheet does not override docs/GROK_BOT_TECH.md or docs/ROSTER.md. Update by adding a line.
