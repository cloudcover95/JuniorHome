# Chief audit

The chief reads. It does not build. It cannot see the other bots. It adapts from the git tip, not from their chat.

## Each 02:00 run
1. Read docs/ROSTER.md.
2. For each roster repo, read the live tip and the commit time.
3. Flag a lane if its repo has had no additive commit in 48 hours.
4. Flag any commit to JuniorSolana or the cloudcover95 profile repo.
5. Flag a push that is not additive (delete, force-push, GitHub Actions).
6. Write one line per flag. No flag means ok.

## Adapt
A flagged lane is told, in chat, to take that repo on its next run. The chief does not commit the fix itself.
No 12-hour skip. One repo per builder run. Bind 127.0.0.1. mint false. socket false. layer_one false.
