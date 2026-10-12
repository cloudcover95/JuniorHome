"""Two gated reward packs. A bot stops the chain."""
from gate_record import close
def run():
    a = close("dev", ["gaia", "home"])
    b = close("home", ["prompt"], a["hash"])
    bot = close("", ["field"], a["hash"])
    return {"ok": a["ok"] and b["ok"] and b["prev"] == a["hash"] and not bot["ok"],
            "first": a["hash"], "second": b["hash"],
            "juniors": round(a["juniors"] + b["juniors"], 4),
            "bot": bot.get("why"), "mint": False}
