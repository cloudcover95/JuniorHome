# Web3 Pipeline Strategy (Ecosystem-Wide)

## Overview
Web3 integration (primarily Solana-focused) is being added across the Junior ecosystem in a sovereign, modular way.

## Core Principles
- No bloat (avoid heavy external services)
- BitNet 1.58 × 3.0 for on-device intelligence
- Swift-adjacent where possible
- Dedicated hardforks for specialized domains (e.g. JuniorSolana)
- Coordinated from JuniorHome

## Current Projects with Web3 Path

| Project       | Web3 Role                          | Status          | Next Action |
|---------------|------------------------------------|-----------------|-------------|
| JuniorHome    | Central coordination + Swift client | Active         | Expand Swift + Solana bridge |
| JuniorSolana  | Dedicated Solana app hardfork     | New (hardfork) | Build app layer + Quick Actions |
| JuniorStock   | Stock/investment intelligence + on-chain | To be wired | Add Web3 module or hardfork |
| JuniorDrive   | Robotics + sim (future on-chain commands) | Planned     | MCP + on-chain hooks |
| BitNet-Intel  | Agentic layer for Web3 decisions  | In progress    | ModelRouter for on-chain tasks |

## Recommended Pattern
1. Core logic stays in JuniorHome / BitNet-Intel
2. Domain-specific apps use hardfork pattern (JuniorSolana style)
3. Python bridge (JuniorPython) for heavy on-chain work
4. Apple Quick Actions / deep linking for mobile experience

## Implementation Status
- JuniorSolana repo created
- Initial architecture docs added
- BitNet-Intel ModelRouter ready for hardware + task routing (including on-chain)

Next: Wire JuniorStock and expand as needed.