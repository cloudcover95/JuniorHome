# Web3 Integration (JuniorPython x JuniorHome)

## Goals
- Swift-adjacent Web3 experience
- Support for ERC20 tokens (original BitNet 1.58 x 3.0 token?)
- Quick Actions / deep linking (Apple style, similar to Coinbase Wallet)
- Python bridge for backend logic

## Proposed Architecture
- Swift app handles UI + wallet connection
- Python (JuniorPython) handles heavy computation + on-chain logic
- BitNet models for on-device intelligence

## Next Steps
- WalletConnect / WalletKit integration
- ERC20 interaction module
- Obsidian-style note + on-chain sync (if relevant)

This keeps everything sovereign while adding Web3 capabilities.