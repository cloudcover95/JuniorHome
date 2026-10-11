"""Traffic share. No mint. No gas. Cloud is a cut."""
CLOUD = 0.05
def share(traffic):
    total = sum(traffic.values())
    if total <= 0:
        return {"ok": False, "why": "no_traffic", "mint": False, "gas": False}
    cloud = round(total * CLOUD, 4)
    rest = total - cloud
    rows = {name: round(rest * n / total, 4) for name, n in traffic.items()}
    return {"ok": True, "rows": rows, "cloud": cloud, "mint": False, "gas": False,
            "pow": False, "same_formula": True, "peers": 0, "network": False}
