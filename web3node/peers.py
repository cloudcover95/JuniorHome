"""Peers are a list. Agreement is a fold over that list."""
def peers(rows):
    seen = []
    for row in rows:
        name = row.get("name")
        if name and name not in seen:
            seen.append(name)
    return {"peers": seen, "n": len(seen)}
def agree(rows):
    book = peers(rows)
    if not book["n"]:
        return {"ok": False, "why": "empty", "peers": [], "consensus": False}
    votes = [bool(r.get("vote")) for r in rows]
    return {"ok": all(votes), "peers": book["peers"], "yes": sum(votes), "n": book["n"],
            "consensus": all(votes), "connected": False}
