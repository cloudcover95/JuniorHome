"""Peer set, vote, majority. No socket."""
def peers(names):
    uniq = []
    for n in names:
        if n and n not in uniq:
            uniq.append(n)
    return {"peers": uniq, "n": len(uniq), "socket": False}
def vote(who, yes, room):
    if who not in room:
        return {"ok": False, "why": "not_peer"}
    return {"ok": True, "who": who, "yes": yes}
def majority(votes, room):
    if len(room) < 1:
        return {"ok": False, "why": "no_peers", "agreed": False}
    yes = sum(1 for v in votes if v.get("yes") and v.get("who") in room)
    need = len(room) // 2 + 1
    return {"ok": yes >= need, "yes": yes, "need": need, "agreed": yes >= need, "socket": False}
