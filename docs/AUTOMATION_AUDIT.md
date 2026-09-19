# Automation audit 2026-09-19 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.5h (not <45 min) — commit allowed. Receipt not rewritten (<24h).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T23 skill-pin tail 19 Sep 01:12–01:17 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | T22 skill-pin last 18 Sep 19:02–19:07 MDT SUCCESS; tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 19 Sep 16:45 in-flight; prior 18 Sep SUCCESS |
| weekly Mon 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 14 Sep 17:20 MDT SUCCESS |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T23 juniorctl skill-pin tail shipped 2026-09-19T01:12-06:00
- bot_slice: T23 juniorctl skill-pin tail (loopback)
- bot_next / next_smallest_slice: T24 juniorctl skill-pin head (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday (2026-09-18 00:00 MDT):
- JuniorLLM: T21 first ~01:22 MDT 18 Sep; T22 last ~19:05 MDT 18 Sep (01:05–01:07 UTC 19 Sep); T23 tail ~01:15–01:17 MDT 19 Sep. Head 9ebf629 feat: T23 juniorctl skill-pin tail CLI + tests (loopback).
- JuniorHome: prior daily audit 18 Sep 16:45 only. Head before this file: 3c093a0 docs: daily audit 2026-09-18 16:45 MDT.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
