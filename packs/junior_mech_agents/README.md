# junior_mech_agents

JuniorHome pack for mech agents on the 45 W node.

Learned from lamm-mit/scientific-agent-skills (Agent Skills layout) and
OpenClaw (workspace-routed agents). Did not vendor their 147 skills.
mistral.rs is an optional local runtime later. AutomataGPT pattern:
structured tokens in, rule-like trit out.

```
scan / CAD dirty / bar ticks
        ↓
AbsMean ternary 6×16 router
        ↓
snap_scan | emit_glb | export_step | fea_flag | stock_intent | halt
```

SolidWorks is an export target (STEP/STL). No COM.

```
python3 -c "import sys; sys.path.insert(0,'packs'); from junior_mech_agents.router import self_test; print(self_test())"
```
