# Automation audit 2026-09-17 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.6h (not <45 min) — commit allowed. Receipt not rewritten (<24h).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T19 skill-pin before HDR 17 Sep 01:03–01:14 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | T18 skill-pin until 16 Sep 19:05–19:10 MDT SUCCESS; tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 17 Sep 16:45 in-flight; prior 16 Sep SUCCESS |
| weekly Sun 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 14 Sep 17:20 MDT SUCCESS |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T19 juniorctl skill-pin before HDR shipped 2026-09-17T01:10-06:00
- bot_slice: T19 juniorctl skill-pin before HDR (loopback)
- bot_next / next_smallest_slice: T20 juniorctl skill-pin after HDR (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday (2026-09-16 00:00 MDT):
- JuniorLLM: T17 since HDR ~01:22 MDT 16 Sep; T18 until HDR ~19:09 MDT 16 Sep; T19 before HDR ~01:12 MDT 17 Sep. Chat after T17: Home imager, Gaia tick, dense tree, ham metrics, trit cache, zt net, imager-auto, operator rails SPIFFE+liboqs, imager_live. Head 4a0763e feat: T19 expose skill_pin_before on juniorctl.
- JuniorHome: chat prod mirrors same window (imager, Gaia tick, tree lurch, ham metric, zt net, operator+persist, imager live) plus prior daily audit 16 Sep 16:45. Head before this file: 3f783db feat: imager live prod.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
