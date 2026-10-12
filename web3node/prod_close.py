"""One entry. Gate, link, one sign, next close."""
from tip_next import next_close
def prod():
    row = next_close()
    if not row["ok"]:
        return {"ok": False, "why": "tip"}
    return {"ok": True, "prev": row["prev"], "hash": row["hash"], "signs": 1,
            "mint": False, "socket": False, "disk": False, "live": False}
