# Automation audit 2026-09-15 16:45 MDT

TZ America/Denver. Owner cloudcover95. One small commit: this file only.
LAST_RECEIPT age ~15.5h (not <45 min) — commit allowed. Receipt not rewritten (<24h).

| slot | taskId prefix | nextRun listed | last known live |
|------|---------------|----------------|-----------------|
| 01:00 overnight | 20718d5d | UI stale (Sep 10 01:00) | T15 skill-pin at HDR 15 Sep 01:14 MDT SUCCESS |
| 07:00 | (not in live list; was 869561c2) | missing | FAIL — automation not listed |
| 13:00 | (not in live list; was d8632910) | missing | FAIL — automation not listed |
| 19:00 | bde8e4e0 | UI stale (Sep 10 19:00) | 14 Sep 19:16 MDT SUCCESS (result title T15 at HDR; git window T14 skill-pin get HEIGHT); tonight 19:00 pending |
| 16:45 daily | dc3fbf2f | listed 12 Sep 16:45 (stale) | this run 15 Sep 16:45 in-flight; prior 14 Sep SUCCESS |
| weekly Sun 17:00 | dd0b1c93 | listed 14 Sep 17:00 (past) | 14 Sep 17:20 MDT SUCCESS |
| weekly Fri 16:45 | 7716afb1 | not in live list | missing / duplicate weekly |

Bot receipt (do not rewrite):
- LAST_RECEIPT: T15 juniorctl skill-pin at HDR shipped 2026-09-15T01:14-06:00
- bot_slice: T15 juniorctl skill-pin at HDR (loopback)
- bot_next / next_smallest_slice: T16 juniorctl skill-pin range FROM TO (loopback)
- chat_slice: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside
- llama_ready: false until JUNIOR_GGUF on box
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

Commits since yesterday:
- JuniorLLM: T14 get HEIGHT ~19:20 MDT 14 Sep; T15 at HDR wrappers ~01:18 MDT 15 Sep. Chat stack 14 Sep (Vulkan/C AbsMean, GGUF map, XYZ, trit harness, agent vote/bench/terraform, OSai engines). Head 3aea798 feat: T15 juniorctl skill_pin_at wrapper (loopback).
- JuniorHome: no 15 Sep commits before this audit. 14 Sep weekly audit 17:22 MDT + daily audit 16:45 MDT + chat prod mirrors. Head before this file: 99f8463 docs: weekly audit 2026-09-14 17:20 MDT.

Blockers:
- GGUF missing (expected); llama_ready false
- 07:00 and 13:00 slots absent from live Automations list
- overnight / 19:00 / daily nextRun UI stale vs actual fires
- weekly Fri 16:45 still missing from live list
