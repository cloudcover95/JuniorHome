# Rails — Swift + future BitNet Linux OS

Implemented in JuniorLLM:
- `rails/swift` JuniorRails Guardrail.swift (prompt scan + private-land publish)
- `rails/linux` bitnetd.service + loopback-only placeholder
- `agent/guardrails.py` skill AST scan, prompt injection, memory topics
- `agent/overnight.py` + `agent/self_prompt.py`
- `obsidian/vault_bridge.py` SKILL.md pin + hash
- `ports/registry.py` BitNet-native vs high-quant vs Fable vs pruned Kimi
- `docs/CLAUDE_FEATURE_MAP.md` + `docs/TEST_ROADMAP_2026.md`

Tests: `PYTHONPATH=. python tests/test_rails_agents.py` — 9/9 pass (2026-08-30).
