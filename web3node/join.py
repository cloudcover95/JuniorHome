"""A node joins by class. No required rack."""
EXAMPLES = ("pi", "gpu", "spark", "mac", "rp2040", "other")
def join(classes):
    seen = []
    for name in classes:
        if name not in seen:
            seen.append(name)
    return {"joined": seen, "n": len(seen), "required": [], "examples": list(EXAMPLES),
            "attached_here": 0, "model_pull": False}
