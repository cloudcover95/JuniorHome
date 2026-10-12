"""Stand guard. Players and supply stay zero."""
from stand import stand
def guard():
    row = stand()
    ok = (row["ok"] and row["players"] == 0 and row["token_supply"] == 0
          and row["mint"] is False and row["socket"] is False
          and row["peers"] == 0 and row["pool"] == "held")
    return {"ok": ok, "players": 0, "supply": 0}
