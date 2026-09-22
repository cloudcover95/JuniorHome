# Automation audit 2026-09-22 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.3h (not <45 min) — commit allowed. Receipt not rewritten (<24h).
STATE.md at-stamp left as 2026-09-22T01:25-06:00 (optional stamp skipped; one commit max).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T29 skill-pin children 22 Sep 01:25–01:32 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | T28 skill-pin child 21 Sep 19:17–19:24 MDT SUCCESS; tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 22 Sep 16:45 in-flight; prior 21 Sep SUCCESS |
| weekly Mon 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 21 Sep 17:25 MDT SUCCESS |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T29 juniorctl skill-pin children shipped 2026-09-22T01:25-06:00
- bot_slice: T29 juniorctl skill-pin children (loopback)
- bot_next / next_smallest_slice: T30 juniorctl skill-pin ancestors (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday (2026-09-21 00:00 MDT):
- JuniorLLM: T27 parent ~01:17–01:19 MDT 21 Sep; T28 child ~19:21–19:23 MDT 21 Sep; T29 children ~01:30–01:32 MDT 22 Sep. Head 1b513c05 feat: T29 tests for juniorctl skill-pin children.
- JuniorHome: daily audit 21 Sep 16:45; weekly audit 21 Sep 17:25. Head before this file: 5fa597c6 docs: weekly audit 2026-09-21 17:25 MDT.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
