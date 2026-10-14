# Automation audit 2026-10-14 ~16:45 MDT (tip 27f8b7a1)

| Name | Owner | Schedule (MDT) | Window | Notes |
|---|---|---|---|---|
| JuniorCloud overnight build | Automations (report only) | daily 01:00 | none (Automations slot) | pre-window guard; before LLC 02:00; outside LLC and personal windows |
| Chief of Staff | LLC | daily 02:00 | LLC 02:00-04:45 | reads only; inside window |
| home-slice (JuniorHome Builder) | LLC | daily 02:15 (`15 2 * * *`) | LLC 02:00-04:45 | enabled, never run; rotates JuniorHome, JuniorDrive, JuniorFetch; Oct 15 = JuniorHome; one file per run; inside window |
| JuniorOS Builder | LLC | daily 02:45 | LLC 02:00-04:45 | inside window |
| Engine Math | LLC | daily 03:15 | LLC 02:00-04:45 | inside window |
| Suite Scripts | LLC | daily 03:45 | LLC 02:00-04:45 | inside window |
| Spatial Suite | LLC | daily 04:15 | LLC 02:00-04:45 | inside window |
| Inference Core | LLC | daily 04:45 | LLC 02:00-04:45 | inside window (window edge) |
| hardware-to-software job | Personal | daily 06:00 | Personal 05:05-07:05 | writes to JuniorHome; inside window |
| JuniorCloud daily audit | Automations (report only) | daily 16:45 | none (Automations slot) | this run; outside both windows |
| JuniorCloud slot 19:00 | Automations (report only) | daily 19:00 | none (Automations slot) | evening guard; outside both windows |
| JuniorCloud weekly audit | Automations (report only) | Mon 17:00 | none (Automations slot) | docs/AUDIT_SLOTS.md says Sun 17:00; conflict; outside both windows |
| omega-obj-weekly | Automations (report only) | Sat 10:00 | none (Automations slot) | outside both windows |
| junioros-core-weekly | Automations (report only) | Sat 11:00 | none (Automations slot) | outside both windows |

Sources: DUAL_AUTONOMY, LLC_GUARDS, ROSTER, LANES, AUDIT_SLOTS, WEEKLY_AUDIT, AUTOMATION_AUDIT. Slots 07:00 and 13:00 are not scheduled.
