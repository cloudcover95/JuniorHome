"""Same fold for a user or a node. A poll is passed in."""
from anticon import fold
def poll(who, role, votes):
    row = fold(who, [role], votes)
    return {"ok": row["ok"], "role": role, "yes": votes.count("yes"), "n": len(votes),
            "why": row.get("why"), "same_fold": True, "scan": False, "network": False}
