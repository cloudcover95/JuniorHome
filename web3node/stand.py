"""Where this stands. Not a live game. Not a token."""
from gate_record_chain import run
def stand():
    row = run()
    return {"ok": row["ok"], "players": 0, "token_supply": 0, "chain": "local",
            "mint": False, "socket": False, "peers": 0, "rate": 0.05, "pool": "held"}
