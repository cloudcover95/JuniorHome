# JuniorStack

Open-source Home suite module. Local catalog, inventory, orders, agent fulfillment.
Ternary pack on short status lines. MIT. JuniorCloud LLC.

## Sale stack
- models/sale_stack.py runs on each order: print job, eco sticker metadata, RFID tag, local Juniors mint receipt
- No printer driver, no RFID writer, no chain broadcast
- Outputs under ~/.juniorhome/juniorstack/prints, tags, and ~/.juniorhome/os/juniors_mint.jsonl

## Integration
- Database at ~/.juniorhome/juniorstack.db
- Agent endpoints on loopback for OSai / LLM
- Local receipts only
