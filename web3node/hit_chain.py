"""Two hit blocks. Second prev is the first hash."""
from hit_block import block
def chain():
    nodes = [{"name": "pi-1", "host": True, "hosts": ["share", "next"]}]
    a = block("share", nodes)
    b = block("next", nodes, a["hash"])
    return {"blocks": [a, b], "linked": b.get("prev") == a.get("hash"),
            "mint": False, "network": False}
