# Weekly audit 2026-09-28 17:04 MDT

TZ America/Denver. Owner cloudcover95. Loopback only. No product work. No deletes.
Window: 2026-09-21 → 2026-09-28.
Daily receipt `docs/AUTOMATION_AUDIT.md` (`c6748c2`, 16:45 MDT) is ~18 min old — still write this file only.

Goals held: local software + BitNet overlay toward JuniorOS; owner public_repos=26 point at Home/JuniorLLM compute; do not clone AbsMean 25 times.

## Automations (live list)

Five active, APP_ONLY. No 07:00 or 13:00 task in the live catalog.

| name | taskId prefix | schedule | nextRun listed (often stale) | enabled |
|------|---------------|----------|------------------------------|---------|
| omega-obj-weekly | 1700c44c | Sat 10:00 Denver | 2026-09-26T16:00Z (fired) | yes |
| JuniorCloud weekly audit | dd0b1c93 | Mon 17:00 Denver | 2026-09-14T23:00Z (this run; UI lag) | yes |
| JuniorCloud daily audit | dc3fbf2f | daily 16:45 Denver | listed 2026-09-12T22:45Z (stale; fires daily) | yes |
| JuniorCloud slot 19:00 | bde8e4e0 | daily 19:00 Denver | listed 2026-09-10T01:00Z (stale) | yes |
| JuniorCloud overnight build | 20718d5d | daily 01:00 Denver | listed 2026-09-10T07:00Z (stale) | yes |

Not in live list (called out by daily audit): slot 07:00 (was 869561c2), slot 13:00 (was d8632910), Friday weekly 16:45 (was 7716afb1).

## Last 7 days — overnight 01:00 (`20718d5d`)

Times below are America/Denver.

| local fire | result title | status |
|------------|--------------|--------|
| 2026-09-28 01:02 | T40 2nd-cousins-once-removed shipped | SUCCESS |
| 2026-09-27 01:02 | T39 skill-pin cousins-once-removed shipped | SUCCESS |
| 2026-09-26 01:11 | T37 skill-pin great-great-grandchildren shipped | SUCCESS |
| 2026-09-25 01:05 | T35 grandchildren shipped | SUCCESS |
| 2026-09-24 01:18 | T33 uncles shipped on main | SUCCESS |
| 2026-09-23 01:17 | T31 siblings shipped on main | SUCCESS |
| 2026-09-22 01:25 | T29 skill-pin children shipped | SUCCESS |
| 2026-09-21 01:13 | T27 skill-pin parent shipped | SUCCESS |

Overnight shipped every night in-window. 2026-09-26 also logged a no-status ghost create at 01:00 then SUCCESS at 01:11. UI `nextRun` lag does not match fires.

## Last 7 days — 07 / 13 / 19 slots

| slot | last 7 days |
|------|-------------|
| 07:00 | FAIL — automation not listed. Zero results. |
| 13:00 | FAIL — automation not listed. Zero results. |
| 19:00 2026-09-27 19:11 | TASK_RESULT_ERROR `USAGE_POOL_EXHAUSTED` |
| 19:00 2026-09-26 19:01 | T38 skill-pin 2nd-cousins shipped SUCCESS |
| 19:00 2026-09-25 19:01 | T36 skill-pin great-grandchildren shipped SUCCESS |
| 19:00 2026-09-24 19:01 | T35 grandchildren slice shipped SUCCESS |
| 19:00 2026-09-23 19:06 | T32 skill-pin cousins shipped SUCCESS |
| 19:00 2026-09-22 19:03 | T30 ancestors shipped SUCCESS |
| 19:00 2026-09-21 19:17 | T28 skill-pin child shipped SUCCESS |
| 19:00 2026-09-28 19:00 | pending (this weekly is 17:04) |

19:00 coverage is the only evening slot that still exists. One quota miss this window (Sep 27). Capacity doc still budgets four slices/day; live catalog delivers two (01:00 + 19:00) plus audits + Saturday omega.

Daily audit `dc3fbf2f` SUCCESS every 16:45 Sep 21–28 (Sep 25 title: GitHub write blocked, audit partial). Prior weekly SUCCESS 2026-09-21 17:25 MDT. omega-obj-weekly SUCCESS 2026-09-26 10:02 MDT.

## GitHub last 7 days

`github___list_commits since=2026-09-21` returned a 100-commit page on JuniorLLM (truncated) and a full Home page. Commit search totals on default branch for author-date 2026-09-21..2026-09-29: JuniorLLM 102, JuniorHome 47. Heads at audit time:

- JuniorLLM `7e6e68f` feat: T40 tests for skill-pin 2nd-cousins-once-removed (2026-09-28 01:10 MDT)
- JuniorHome `c6748c2` docs: daily audit 2026-09-28 16:45 MDT; chat/prod cluster today is JuniorDeck (JDB1 / SMF / wavetable / Csound / LMMS+Qtractor probes)

Public repo count on owner: 26 (was 25 last weekly). Compute stays pointed at Home + JuniorLLM. Do not vendor AbsMean per-repo.

### T-slices (bot / overnight+19:00)

T4 ledger and T5 skill pin remain **done** (prior weeks). This week the bot walked T28→T40 on the skill-pin kinship stack:

| slice | where |
|-------|--------|
| T4 JuniorFileLedger create/read | backlog: done (prior) |
| T5 skill load + hash pin | backlog: done (2026-09-12) |
| T27 skill-pin parent | overnight 2026-09-21 SUCCESS |
| T28 skill-pin child | 19:00 2026-09-21 SUCCESS |
| T29 skill-pin children | overnight 2026-09-22 SUCCESS |
| T30 ancestors | 19:00 2026-09-22 SUCCESS |
| T31 siblings | overnight 2026-09-23 SUCCESS |
| T32 cousins | 19:00 2026-09-23 SUCCESS |
| T33 uncles | overnight 2026-09-24 SUCCESS |
| T35 grandchildren | overnight + 19:00 2026-09-24/25 SUCCESS |
| T36 great-grandchildren | 19:00 2026-09-25 SUCCESS |
| T37 great-great-grandchildren | overnight 2026-09-26 SUCCESS |
| T38 2nd-cousins | 19:00 2026-09-26 SUCCESS |
| T39 first-cousins-once-removed | overnight 2026-09-27 SUCCESS |
| T40 2nd-cousins-once-removed | overnight 2026-09-28 SUCCESS |
| T41 skill-pin third-cousins | **next_smallest_slice** (open) |

Port on receipts: JuniorAstraReason. LAST_RECEIPT file still reads T39 @ 2026-09-27T01:06-06:00 (stale vs T40 head) — not rewritten here.

### Chat slices (Home UI, user/app/media/scan, TP)

STATE.md `chat_slice`: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside.

Chat/prod on JuniorHome this week (not T41): JuniorDeck session/JACK/JDB1 70 B + SMF 138 B + music_map + wavetable + Csound host + LMMS/Qtractor probes; omega Blender 4.2 ext + OBJ writer (ue5_launch false); Gaia/qutrit/Gell-Mann/trit-gate prod pointers; shop-lab / ROS2 DDS / OSai IoT protocol box; Climbs Gaia port probe (pollinate only). Daily `AUTOMATION_AUDIT.md` rewrite each 16:45. JuniorLLM bot commits stay on juniorctl skill-pin CLI + tests (loopback) plus omega blender_ext.

## Shipped

- T28–T40 skill-pin kinship chain on JuniorLLM rails/linux juniorctl (loopback, no body/fetch/exec).
- Overnight 01:00 SUCCESS 8/8 in-window; 19:00 SUCCESS 6/7 completed fires (1 quota miss).
- Daily AUTOMATION_AUDIT table current through 16:45 MDT today (`c6748c2`).
- omega-obj-weekly: blender_ext + ports/blender_omega (Sat 2026-09-26).
- Home chat/prod pointers listed above (JuniorDeck cluster 2026-09-28).
- This WEEKLY_AUDIT.md week-3 header (prior 2026-09-21 and 2026-09-14 sections kept below).

## Skipped

- Recreate 07:00 / 13:00 automations (audit does not create tasks).
- T41 skill-pin third-cousins (left for overnight/19:00).
- Rewrite LAST_RECEIPT / STATE.md / AUTOMATION_AUDIT.md (daily receipt <45 min; this file only).
- Any GGUF download, AbsMean clone farm, docker.sock, 0.0.0.0 bind, MP/KAYA, force-push, delete.
- Tonight 19:00 (not due yet).

## Gaps

1. **llama GGUF** — `llama_ready: false until JUNIOR_GGUF on box`. Map + T3 / onboard notes exist; no on-box weight. Expected.
2. **19:00 coverage** — slot exists and shipped 6/7 evenings; Sep 27 `USAGE_POOL_EXHAUSTED`; UI nextRun stale; 07/13 still missing so the 4-slot capacity plan is 50% live. Tonight 19:00 not yet fired.
3. **CI emails vs local test_prod** — notification is APP_ONLY (no CI email trail). Code search `test_prod` on JuniorLLM + JuniorHome returned zero hits this run. Local proof stays on-box tests named in receipts (`test_t39_...` / `test_t40_...`), not a mailed CI gate.
4. Stale `nextRun` on overnight / 19:00 / daily / weekly vs actual SUCCESS fires.
5. LAST_RECEIPT lag: file still T39 while STATE + overnight already stamped T40.
6. Owner public_repos now 26 vs stated 25-repo target — still do not vendor AbsMean per-repo.

## Receipt stamps (do not rewrite)

- LAST_RECEIPT file: T39 juniorctl skill-pin first-cousins-once-removed shipped 2026-09-27T01:06-06:00
- STATE at: 2026-09-28T01:02-06:00 bot_slice T40 / bot_next T41
- llama_ready: false
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

---

# Weekly audit 2026-09-21 17:25 MDT

TZ America/Denver. Owner cloudcover95. Loopback only. No product work. No deletes.
Window: 2026-09-14 → 2026-09-21.
Daily receipt `docs/AUTOMATION_AUDIT.md` (`ad1ee18`, 16:45 MDT) is ~40 min old — still write this file only.

Goals held: local software + BitNet overlay toward JuniorOS; 25 public repos point at Home/JuniorLLM compute; do not clone AbsMean 25 times.

## Automations (live list)

Four active, APP_ONLY. No 07:00 or 13:00 task in the live catalog.

| name | taskId prefix | schedule | nextRun listed (often stale) | enabled |
|------|---------------|----------|------------------------------|---------|
| JuniorCloud weekly audit | dd0b1c93 | Mon 17:00 Denver | 2026-09-14T23:00Z (this run; UI lag) | yes |
| JuniorCloud daily audit | dc3fbf2f | daily 16:45 Denver | listed 2026-09-12T22:45Z (stale) | yes |
| JuniorCloud slot 19:00 | bde8e4e0 | daily 19:00 Denver | listed 2026-09-10T01:00Z (stale) | yes |
| JuniorCloud overnight build | 20718d5d | daily 01:00 Denver | listed 2026-09-10T07:00Z (stale) | yes |

Not in live list (called out by daily audit): slot 07:00 (was 869561c2), slot 13:00 (was d8632910), Friday weekly 16:45 (was 7716afb1).

## Last 7 days — overnight 01:00 (`20718d5d`)

Times below are America/Denver.

| local fire | result title | status |
|------------|--------------|--------|
| 2026-09-21 01:13 | T27 skill-pin parent shipped | SUCCESS |
| 2026-09-20 01:18 | T25 skill-pin count shipped | SUCCESS |
| 2026-09-19 01:12 | T23 skill-pin tail shipped | SUCCESS |
| 2026-09-18 01:19 | T21 skill-pin first shipped | SUCCESS |
| 2026-09-17 01:03 | T19 skill-pin-before-HDR shipped | SUCCESS |
| 2026-09-16 01:18 | T17 skill-pin since HDR shipped | SUCCESS |
| 2026-09-15 01:14 | T15 skill-pin at HDR shipped | SUCCESS |
| 2026-09-14 01:15 | T13 skill-pin height shipped | SUCCESS |

Overnight shipped every night in-window. UI `nextRun` lag does not match fires.

## Last 7 days — 07 / 13 / 19 slots

| slot | last 7 days |
|------|-------------|
| 07:00 | FAIL — automation not listed. Zero results. |
| 13:00 | FAIL — automation not listed. Zero results. |
| 19:00 2026-09-20 19:06 | T26 skill-pin genesis shipped SUCCESS |
| 19:00 2026-09-19 19:00 | T24 skill-pin head shipped SUCCESS |
| 19:00 2026-09-18 19:02 | T22 skill-pin last shipped SUCCESS |
| 19:00 2026-09-17 19:16 | T21 skill-pin first shipped SUCCESS |
| 19:00 2026-09-16 19:05 | T18 skill-pin until shipped SUCCESS |
| 19:00 2026-09-15 19:10 | T16 skill-pin range shipped SUCCESS |
| 19:00 2026-09-14 19:16 | T15 skill-pin at HDR shipped SUCCESS |
| 19:00 2026-09-21 19:00 | pending (this weekly is 17:25) |

19:00 coverage is the only evening slot that still exists. All in-window 19:00 fires SUCCESS (quota miss Sep 12 is outside this window). Capacity doc still budgets four slices/day; live catalog delivers two (01:00 + 19:00) plus audits.

Daily audit `dc3fbf2f` SUCCESS every 16:45 Sep 14–21. Prior weekly SUCCESS 2026-09-14 17:20 MDT (`WEEKLY_AUDIT.md` first write).

## GitHub last 7 days

`github___list_commits since=2026-09-14` returned a 100-commit page each. Commit search totals on default branch for author-date 2026-09-14..2026-09-22: JuniorLLM 171, JuniorHome 135. Heads at audit time:

- JuniorLLM `773b80d` feat: T27 juniorctl skill_pin_parent bind (2026-09-21 01:19 MDT)
- JuniorHome `ad1ee18` docs: daily audit 2026-09-21 16:45 MDT; chat/prod cluster peaked 2026-09-16 (imager / operator / Gaia / pq / OSai pointers)

Public repo count on owner: 25. Compute stays pointed at Home + JuniorLLM. Do not vendor AbsMean per-repo.

### T-slices (bot / overnight+19:00)

T4 ledger and T5 skill pin remain **done** (prior weeks). This week the bot walked T13→T27 on the skill-pin stack:

| slice | where |
|-------|--------|
| T4 JuniorFileLedger create/read | backlog: done (prior) |
| T5 skill load + hash pin | backlog: done (2026-09-12) |
| T13 skill-pin height | overnight 2026-09-14 SUCCESS |
| T15 skill-pin at HDR | overnight 2026-09-15 + 19:00 2026-09-14 SUCCESS |
| T16 skill-pin range | 19:00 2026-09-15 SUCCESS |
| T17 skill-pin since HDR | overnight 2026-09-16 SUCCESS |
| T18 skill-pin until | 19:00 2026-09-16 SUCCESS |
| T19 skill-pin before HDR | overnight 2026-09-17 SUCCESS |
| T21 skill-pin first | overnight 2026-09-18 + 19:00 2026-09-17 SUCCESS |
| T22 skill-pin last | 19:00 2026-09-18 SUCCESS |
| T23 skill-pin tail | overnight 2026-09-19 SUCCESS |
| T24 skill-pin head | 19:00 2026-09-19 SUCCESS |
| T25 skill-pin count | overnight 2026-09-20 SUCCESS |
| T26 skill-pin genesis | 19:00 2026-09-20 SUCCESS |
| T27 skill-pin parent | overnight 2026-09-21 SUCCESS |
| T28 skill-pin child | **next_smallest_slice** (open) |

Port on receipts: JuniorAstraReason. LAST_RECEIPT `2026-09-21T01:13-06:00` — not rewritten here.

### Chat slices (Home UI, user/app/media/scan, TP)

STATE.md `chat_slice`: Home UI + user/app/media/scan + TP + BitnetCloud + llama sit-beside.

Chat/prod on JuniorHome this week (not T28): imager live/auto + tree zoom/lurch; operator box status + persist views; Gaia autonomy tick; zt net + secure/semantic cache; ham/pq + ML-KEM status; trit latch bench; winsor/sparsity/OSai/AGI gates; GGUF onboard note without a box weight. Daily `AUTOMATION_AUDIT.md` rewrite each 16:45. JuniorLLM bot commits stay on juniorctl skill-pin CLI + tests (loopback).

## Shipped

- T13–T27 skill-pin chain on JuniorLLM rails/linux juniorctl (loopback, no body/fetch/exec).
- Overnight 01:00 SUCCESS 8/8 in-window; 19:00 SUCCESS 7/7 completed fires.
- Daily AUTOMATION_AUDIT table current through 16:45 MDT today (`ad1ee18`).
- Home chat/prod pointers listed above (Sep 16 cluster).
- This WEEKLY_AUDIT.md week-2 header (prior 2026-09-14 section kept below).

## Skipped

- Recreate 07:00 / 13:00 automations (audit does not create tasks).
- T28 skill-pin child (left for overnight/19:00).
- Rewrite LAST_RECEIPT / STATE.md / AUTOMATION_AUDIT.md (daily receipt <45 min; this file only).
- Any GGUF download, AbsMean clone farm, docker.sock, 0.0.0.0 bind, MP/KAYA, force-push, delete.
- Tonight 19:00 (not due yet).

## Gaps

1. **llama GGUF** — `llama_ready: false until JUNIOR_GGUF on box`. Map + T3 / onboard notes exist; no on-box weight. Expected.
2. **19:00 coverage** — slot exists and shipped every in-window evening; UI nextRun stale; 07/13 still missing so the 4-slot capacity plan is 50% live. Tonight 19:00 not yet fired.
3. **CI emails vs local test_prod** — notification is APP_ONLY (no CI email trail). Code search `test_prod` on JuniorLLM + JuniorHome returned zero hits this run. Local proof stays on-box tests named in receipts (`test_t27_...`), not a mailed CI gate.
4. Stale `nextRun` on overnight / 19:00 / daily / weekly vs actual SUCCESS fires.
5. Friday weekly duplicate (7716afb1) absent; Mon 17:00 is the live weekly.

## Receipt stamps (do not rewrite)

- LAST_RECEIPT: T27 juniorctl skill-pin parent shipped 2026-09-21T01:13-06:00
- bot_next: T28 juniorctl skill-pin child (loopback)
- llama_ready: false
- bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd
- port: JuniorAstraReason

---

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
