# Ecosystem Structure Cleanup (June 2026)

## Current Official Repos

- **JuniorHome** — Central hub + Swift client + deployment
- **JuniorAGI_SDK** — Core logic layer (WorkflowEngine, ModelRouter, Web3Agent, etc.)
- **JuniorSOL** — Solana Web3 app (correct name)
- **JuniorDrive** — Robotics, driving sim, VR/AR
- **JuniorStock**, **JuniorPiPython**, etc. — Domain-specific as needed

## Cleanup Actions Taken
- Logic from temporary AGI_SDK moved into JuniorAGI_SDK
- JuniorSOL created with correct naming
- References updated

All future development will respect existing repo names and consolidate logic into JuniorAGI_SDK where appropriate.