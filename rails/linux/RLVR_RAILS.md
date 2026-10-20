# JuniorOS production rails — RLVR

Local only. No model pull.

## Workflow
scripts/home_rlvr.py reads the latest cycle from:
- ~/.juniorhome/os/rlvr.jsonl
- ~/.juniorhome/os/rlvr_trit.jsonl

It writes a status tile to:
- ~/.juniorhome/os/home_rlvr.json

## End user
```bash
python scripts/rlvr_cli.py --show
```

Shows the latest envelope layer, hardware, and trit score if present.

## Handoff
If the trit score is low, the envelope is T0 and handoff is marked repack.
Home rails can surface that without running the agent.

Additive. MIT.
