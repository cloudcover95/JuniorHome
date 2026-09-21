# Automation audit 2026-09-21 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.5h (not <45 min) — commit allowed. Receipt not rewritten (<24h).
STATE.md at-stamp left as 2026-09-21T01:13-06:00 (optional stamp skipped; one commit max).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T27 skill-pin parent 21 Sep 01:13–01:19 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | T26 skill-pin genesis 20 Sep 19:06–19:11 MDT SUCCESS; tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 21 Sep 16:45 in-flight; prior 20 Sep SUCCESS |
| weekly Mon 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 14 Sep 17:20 MDT SUCCESS; due today 17:00 |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T27 juniorctl skill-pin parent shipped 2026-09-21T01:13-06:00
- bot_slice: T27 juniorctl skill-pin parent (loopback)
- bot_next / next_smallest_slice: T28 juniorctl skill-pin child (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday (2026-09-20 00:00 MDT):
- JuniorLLM: T25 count ~01:21–01:23 MDT 20 Sep; T26 genesis ~19:09–19:11 MDT 20 Sep; T27 parent ~01:17–01:19 MDT 21 Sep. Head 773b80d feat: T27 juniorctl skill_pin_parent bind.
- JuniorHome: prior daily audit 20 Sep 16:45 only. Head before this file: 43bb79d docs: daily audit 2026-09-20 16:45 MDT.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
