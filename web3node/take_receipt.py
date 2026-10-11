"""A presented take gets a receipt. Cold does not."""
import hashlib
from cold_take import take_cold
def receipt(actor, hits, cuts, cold):
    row = take_cold(actor, hits, cuts, cold)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "hash": None}
    body = f"{actor}|{row['yield']}".encode()
    return {"ok": True, "hash": hashlib.sha256(body).hexdigest()[:16], "yield": row["yield"]}
