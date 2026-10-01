# Automation audit 2026-09-30 22:14 MDT

Live, app-only, America/Denver.

| Name | Id | Cadence | Prompt state |
|------|----|---------|----------------|
| omega-obj-weekly | 1700c44c | Sat 10:00 | OBJ check, ue5_launch false |
| JuniorCloud weekly audit | dd0b1c93 | Mon 17:00 | WEEKLY_AUDIT.md only |
| JuniorCloud daily audit | dc3fbf2f | daily 16:45 | AUTOMATION_AUDIT.md |
| JuniorCloud slot 19:00 | bde8e4e0 | daily 19:00 | one additive slice |
| JuniorCloud overnight build | 20718d5d | daily 01:00 | one additive slice |

Missing from the live list: 07:00 and 13:00 slots named inside the daily prompt.
No event triggers. No 0.0.0.0. No model pull in these prompts.

Next: ecosystem port check lives in JuniorLLM scripts/ecosystem_ports_prod.py.
