# Bot first

Work happens in the Grok bot. Git is the last step. cloudcover95. Additive only.
Maximize the bot turn. Minimize the git touch. One shallow clone. One push. No mid-turn fetch.

## Order
1. The bot drafts the file in its own turn. No git yet.
2. The bot self-checks against the standing rules. No git yet.
3. The bot sends a one-line pre-push report to its chief. No git yet.
4. The bot does one shallow clone of the one repo it writes to.
5. The bot adds the one new file.
6. The bot pushes once.
7. The bot sends the sha.

If the self-check fails, stop. Do not clone. Do not push.
If the tip is recent, take a different path. Do not fetch again to confirm.

## What is not allowed
- No fetch before the draft is done.
- No second fetch to check the tip.
- No status, log, or diff against origin mid-turn.
- No clone of a repo the bot does not write to.
- No download of guard_preflight.py or any remote script.
- No pre-commit. No ruff. No mypy.
- No waiting for an okay after the one-line report. Report, then push.

## Detail
Only detail when high level is required. A pass is one line: repo, path, sha. A fail is one line: what failed. No dump of the file. No dump of the diff.

## Both accounts
The LLC bots and the personal quant team use the same order. The chief on each side collects the one-line reports. The verifier on the personal side collects the shas. Neither chief nor verifier clones a repo to check a tip. They read the tip once, at the start, and stop.

## Status
Live. Does not override docs/DUAL_AUTONOMY.md, docs/PERSONAL_TEAM_BRIDGE.md, or docs/PERSONAL_QUANT_TEAM.md. Update by adding a line.
