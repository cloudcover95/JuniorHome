# Tuner ledger

Local append-only log, vault receipt, brain note. JuniorHome. Additive only. No flash. No model. No network.

## Files
- web3node/tuner/ledger.py → ~/.juniorhome/gaia_mesh/tuner.jsonl
- web3node/tuner/vault.py → ~/.juniorhome/vault/tuner/<close>.json
- web3node/tuner/brain.py → ~/.juniorhome/brain/tuner.jsonl

## Close
sha256(prev || pack || text)[:16]. Rate 0.05.

## Status
Live. Pair with docs/TUNER.md.
