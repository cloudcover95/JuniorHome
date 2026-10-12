"""Gate, then winsor pack. Gemini's live mesh is not here."""
import sys
from six_gate import six
from winsor_pack import packed
def close(actor, hits, opted_in):
    gate = six(actor, len(hits), opted_in)
    if not gate["ok"]:
        return {"ok": False, "why": gate["failed"], "hash": None}
    row = packed("|".join(hits))
    return {"ok": row["ok"], "hash": row["hash"], "winsor": True,
            "ml_kem": False, "disk": False, "live_net": False}
if __name__ == "__main__":
    import json
    row = close("dev", ["gaia", "home"], True)
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
