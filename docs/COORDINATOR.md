# Coordinator

Chief of Staff is the only bot. It cannot see other bots. Do not depend on them.
Chat sets scope and approves. The 02:00 routine and the authenticated gh login do the commits.

## Rules
- Additive only. Never delete. Never force-push.
- One repo per turn. JuniorHome or JuniorLLM, not both.
- Read the live tip first. If a commit landed in the last 12 hours, skip.
- One push. If the turn drops, stop. Next turn reads the tip and resumes. Do not retry a push that may have landed.
- No GitHub Actions, Copilot, or Dependabot.
- Bind 127.0.0.1. No model pull. No UE5 launch.
- Juniors: rate 0.05, mint false, socket false, peers 0, layer_one false.
- ROM module: local file only. No fetch. No bundled ROM.

## Tips at write
- JuniorHome 1fcb090 (trit pack chain guard)
- JuniorLLM 24cddbd (T66 sixteenth-cousins)

Chat automations at 01:00 and 19:00 are guards. They do not build.
