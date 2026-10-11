"""A link from outside the room is denied. No socket."""
from peers import peers
def link(who, room, prev, parent):
    names = peers(room)["peers"]
    if who not in names:
        return {"ok": False, "why": "not_peer", "linked": False, "socket": False}
    if prev != parent:
        return {"ok": False, "why": "prev", "linked": False, "socket": False}
    return {"ok": True, "who": who, "linked": True, "socket": False}
