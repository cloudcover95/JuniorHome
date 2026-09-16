# Automation audit 2026-09-16 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.3h (not <45 min) — commit allowed. Receipt not rewritten (<24h).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T17 skill-pin since HDR 16 Sep 01:18–01:24 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | T16 skill-pin range 15 Sep 19:10–19:16 MDT SUCCESS; tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 16 Sep 16:45 in-flight; prior 15 Sep SUCCESS |
| weekly Sun 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 14 Sep 17:20 MDT SUCCESS |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T17 juniorctl skill-pin since HDR shipped 2026-09-16T01:25-06:00
- bot_slice: T17 juniorctl skill-pin since HDR (loopback)
- bot_next / next_smallest_slice: T18 juniorctl skill-pin until HDR (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday (2026-09-15 00:00 MDT):
- JuniorLLM: T17 since HDR ~01:22 MDT 16 Sep (a3d4438 / 5763fc2 / e9b19e4). Chat after T17: trit/SOL/Hamming/PQ/ML-KEM, OSai goldens, Home imager, Gaia tick, dense tree, ham metrics. Head 12eb9eb feat: normalized bit Hamming metrics.
- JuniorHome: chat prod mirrors same window (imager, Gaia tick, tree lurch, ham metric). Head before this file: 2a61acd feat: ham metric prod.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
