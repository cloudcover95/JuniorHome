import json
from pathlib import Path
from community_saas import credit
from ledger_verify import verify
META = {"name": "JuniorOS Credit", "symbol": "JOS", "decimals": 0, "chain": None}
def stamp():
    row = credit("cu-aurora", 0.0, "erc-meta")
    return {**META, "ledger": verify(), "onchain": False, "tip": row["hash"]}
