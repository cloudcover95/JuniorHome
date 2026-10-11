"""What a block actually holds."""
from close_sign import sign
def record(hits, user):
    row = sign(hits)
    return {"user": user, "holds": ["hit counts", "close hash", "one signature"],
            "does_not_hold": ["content body", "peer list", "balance"],
            "community": "cloud cut of the same formula", "mint": False,
            "identity": "address is not the signer", "sig": row["sig"],
            "signs": row["signs"], "network": False}
