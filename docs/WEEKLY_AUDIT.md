# Weekly audit 2026-09-14 17:20 MDT

TZ America/Denver. Owner cloudcover95. Loopback only. No product work. No deletes.
Window: 2026-09-07 → 2026-09-14.
Daily receipt `docs/AUTOMATION_AUDIT.md` (`7c478b3`, 16:45 MDT) is ~35 min old — still write this file only.

Goals held: local software + BitNet overlay toward JuniorOS; 25 public repos point at Home/JuniorLLM compute; do not clone AbsMean 25 times.

## Automations (live list)

Four active, APP_ONLY. No 07:00 or 13:00 task in the live catalog.

| name | taskId prefix | schedule | nextRun listed (often stale) | enabled |
|------|---------------|----------|------------------------------|---------|
| JuniorCloud weekly audit | dd0b1c93 | Sun 17:00 Denver | 2026-09-14T23:00Z (this run) | yes |
| JuniorCloud daily audit | dc3fbf2f | daily 16:45 Denver | listed 2026-09-12T22:45Z (stale) | yes |
| JuniorCloud slot 19:00 | bde8e4e0 | daily 19:00 Denver | listed 2026-09-10T01:00Z (stale) | yes |
| JuniorCloud overnight build | 20718d5d | daily 01:00 Denver | listed 2026-09-10T07:00Z (stale) | yes |

Not in live list (called out by daily audit): slot 07:00 (was 869561c2), slot 13:00 (was d8632910), Friday weekly 16:45 (was 7716afb1).

## Last 7 days — overnight 01:00 (`20718d5d`)

Times below are America/Denver.

| local fire | result title | status |
|------------|--------------|--------|
| 2026-09-14 01:15 | T13 skill-pin height shipped | SUCCESS |
| 2026-09-13 01:13 | T9 skill-pin verify shipped | SUCCESS |
| 2026-09-12 01:07 | T5 skill load + hash pin shipped | SUCCESS |
| 2026-09-11 01:07 | C7 overlay PATH pin shipped | SUCCESS |
| 2026-09-10 01:12 | A7 exports shipped to main | SUCCESS |
| 2026-09-09 ~03:11–03:26 | overlay installer / A8 evals (manual cluster) | SUCCESS |

Overnight shipped every night in-window. UI `nextRun` lag does not match fires.

## Last 7 days — 07 / 13 / 19 slots

| slot | last 7 days |
|------|-------------|
| 07:00 | FAIL — automation not listed. Zero results. |
| 13:00 | FAIL — automation not listed. Zero results. |
| 19:00 2026-09-13 19:16 | T12 skill-pin log shipped SUCCESS |
| 19:00 2026-09-12 19:07 | T8 skill-pin pin shipped SUCCESS |
| 19:00 2026-09-11 19:17 | TASK_RESULT_ERROR `USAGE_POOL_EXHAUSTED` |
| 19:00 2026-09-10 19:11 | C6 OCI bundle shipped SUCCESS |
| 19:00 2026-09-09 19:27 | B5 Kimi K3 edge shipped SUCCESS |
| 19:00 2026-09-14 19:00 | pending (this weekly is 17:20) |

19:00 coverage is the only evening slot that still exists. One quota miss (Sep 11). Capacity doc still budgets four slices/day; live catalog delivers two (01:00 + 19:00) plus audits.

Daily audit `dc3fbf2f` SUCCESS on Sep 11–14 (16:45). Weekly has no prior results; this is the first write of `WEEKLY_AUDIT.md`.

## GitHub last 7 days

`github___list_commits since=2026-09-07` returned a 100-commit page each. Commit search totals on default branch for the same window: JuniorLLM 288, JuniorHome 196. Heads at audit time:

- JuniorLLM `4130ba5` feat: three OSai local-train engines (2026-09-14 14:40 MDT)
- JuniorHome `7c478b3` docs: daily automation audit 2026-09-14 16:45 MDT; prior product head `f5c7719` feat: osai engines prod

Public repo count on owner: 25. Compute stays pointed at Home + JuniorLLM. Do not vendor AbsMean per-repo.

### T-slices (bot / overnight+19:00)

T4 ledger and T5 skill pin are **done** (JuniorLLM `docs/COMPILED_BACKLOG.md`). This week the bot walked the skill-pin stack, not a new T4:

| slice | where |
|-------|--------|
| T4 JuniorFileLedger create/read | backlog: done (prior) |
| T5 skill load + hash pin | overnight 2026-09-12 SUCCESS |
| T8 skill-pin pin | 19:00 2026-09-12 SUCCESS |
| T9 skill-pin verify | overnight 2026-09-13 SUCCESS |
| T10 verify-one | chat-hours 2026-09-13 ~07:35 MDT (JuniorLLM) |
| T11 skill-pin tip | 2026-09-13 ~13:09 MDT |
| T12 skill-pin log | 19:00 2026-09-13 SUCCESS |
| T13 skill-pin height | overnight 2026-09-14 SUCCESS |
| T14 skill-pin get HEIGHT | **next_smallest_slice** (open) |

Port on receipts: JuniorAstraReason. LAST_RECEIPT `2026-09-14T01:15-06:00` — not rewritten here.

Home also logged T4-adjacent fused-list / ORT-off notes (chat, not the T4 ledger slice).

### Chat slices (Home UI, user/app/media/scan, TP)

STATE.md `chat_slice`: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside.

Chat-heavy this week (not T14): Home dash / StoneField / Gaia handshake; Vulkan + C AbsMean/Winsor (optional compile, one copy — not 25 clones); GGUF map + T3 header without a box weight; XYZ IQ1_S/TQ1; trit harness; agent vote/bench/terraform; OSai local-train engines; Asahi/JuniorOS goldens. JuniorHome mirrored as prod/docs pointers.

## Shipped

- T5–T13 skill-pin chain on JuniorLLM rails/linux juniorctl (loopback, no body/fetch/exec).
- C6 OCI bundle + C7 overlay PATH pin (JuniorOS overlay track).
- A7 exports, A8 evals (start of window).
- Daily AUTOMATION_AUDIT table current through 16:45 MDT today.
- Chat stack listed above, pointers only on Home.
- First WEEKLY_AUDIT.md (this file).

## Skipped

- Recreate 07:00 / 13:00 automations (audit does not create tasks).
- T14 get HEIGHT (left for overnight/19:00).
- Rewrite LAST_RECEIPT / STATE.md (daily receipt <45 min; receipt itself ~16h old and next slice still open).
- Any GGUF download, AbsMean clone farm, docker.sock, 0.0.0.0 bind, MP/KAYA, force-push, delete.
- Tonight 19:00 (not due yet).

## Gaps

1. **llama GGUF** — `llama_ready: false until JUNIOR_GGUF on box`. Map + T3 header exist; no on-box weight. Expected.
2. **19:00 coverage** — slot exists and mostly ships, but Sep 11 `USAGE_POOL_EXHAUSTED`; UI nextRun stale; 07/13 still missing so the 4-slot capacity plan is 50% live.
3. **CI emails vs local test_prod** — notification is APP_ONLY (no CI email trail). No `test_prod` hit in JuniorLLM/JuniorHome code search this run. Local proof stays on-box tests named in receipts (`test_t13_...`), not a mailed CI gate.
4. Stale `nextRun` on overnight / 19:00 / daily vs actual SUCCESS fires.
5. Friday weekly duplicate (7716afb1) absent; Sun 17:00 is the live weekly.

## Receipt stamps (do not rewrite)

- LAST_RECEIPT: T13 juniorctl skill-pin height shipped 2026-09-14T01:15-06:00
- bot_next: T14 juniorctl skill-pin get HEIGHT (loopback)
- llama_ready: false
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason
