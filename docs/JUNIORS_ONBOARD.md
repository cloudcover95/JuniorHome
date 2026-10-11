# Juniors onboard

JuniorHome is the hit environment. This is not the PiThon handshake.

1. `python3 scripts/onboard_prod.py` makes one hosted hit and a receipt.
2. `python3 scripts/hit_block_prod.py` packs that receipt into a trit block.
3. `python3 scripts/hit_sign_prod.py` signs the tip with ML-DSA-65.

No mint. No gas. No network scan. A node earns only if it hosts content that was hit.
