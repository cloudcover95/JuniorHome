# Automation audit 2026-09-18 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.4h (not <45 min) — commit allowed. Receipt not rewritten (<24h).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T21 skill-pin first 18 Sep 01:19–01:24 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | T20 skill-pin after HDR 17 Sep 19:16–19:21 MDT SUCCESS (result title said T21; commits were T20); tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 18 Sep 16:45 in-flight; prior 17 Sep SUCCESS |
| weekly Mon 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 14 Sep 17:20 MDT SUCCESS |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T21 juniorctl skill-pin first shipped 2026-09-18T01:19-06:00
- bot_slice: T21 juniorctl skill-pin first (loopback)
- bot_next / next_smallest_slice: T22 juniorctl skill-pin last (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday (2026-09-17 00:00 MDT):
- JuniorLLM: T19 before HDR ~01:12 MDT 17 Sep; T20 after HDR ~19:19 MDT 17 Sep; T21 first ~01:22 MDT 18 Sep. Head 15c2bad feat: T21 juniorctl skill_pin_first export.
- JuniorHome: prior daily audit 17 Sep 16:45 only. Head before this file: 32d3207 docs: daily audit 2026-09-17 16:45 MDT.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
