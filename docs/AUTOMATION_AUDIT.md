# Automation audit 2026-10-01 16:45 MDT

| Name | When | Job |
|------|------|-----|
| overnight | daily 01:00 | one additive slice |
| slot 19:00 | daily 19:00 | one additive slice |
| daily audit | daily 16:45 | AUTOMATION_AUDIT.md |
| weekly audit | Mon 17:00 | WEEKLY_AUDIT.md |
| omega-obj-weekly | Sat 10:00 | OBJ, ue5_launch false |
| junioros-core-weekly | Sat 11:00 | ecosystem.py + os_tick |

07:00 and 13:00 are not scheduled. App-only. America/Denver. Loopback only.

| Check | Result |
|------|--------|
| ports/ecosystem.py | present, 34 suite cores, blob f4623b99 |
| scripts/suite_tick_prod.py | JuniorHome present, blob 2a7a26ac; JuniorLLM path absent |
| overlap | LAST_RECEIPT 2026-10-01T01:02-06:00, older than 45 min |
