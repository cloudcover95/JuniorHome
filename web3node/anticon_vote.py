"""Node or user. AntiCon before the trit majority."""
from anticon import fold
from trit_vote import trit_vote
def secure(who, role, room, yes):
    gate = fold(who, [role], yes)
    if not gate["ok"]:
        return {"ok": False, "why": gate["why"], "agreed": False, "role": role}
    row = trit_vote(room, yes)
    row["role"] = role
    row["who"] = who
    return row
