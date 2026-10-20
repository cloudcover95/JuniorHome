# JuniorStack

Open-source local ops module for JuniorHome and JuniorOS.

General catalog, inventory, orders, and agent fulfillment.
Not a specific business. Lanes and products are user-defined.
Local-first. Ternary packing on status lines where it fits.
Spins up as a Home suite module.

## What it is

- Catalog and admin over loopback
- SQLite at ~/.juniorhome/juniorstack.db
- Local invoice, receipt, and message logs
- Agent endpoints for address check and fulfillment start
- Status digest for JuniorHome rails

## Run

```bash
cd packs/juniorstack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/seed.py
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## License

MIT. JuniorCloud LLC.
