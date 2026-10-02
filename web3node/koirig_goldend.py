"""Surface flip across Koirig and Goldend. Energy is the trit score."""
PROTOCOL = "goldend-osai-omega/1"
LAND = ("note", "step", "gamma", "filter", "clock", "mix")
SEA = ("tick", "gate", "envelope")
def energy(text):
    xs = [((ord(c) % 5) - 2) / 2.0 for c in (text or "x")[:32]]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    zeros = 0
    for x in xs:
        q = round(x / gamma)
        t = 1 if q > 1 else (-1 if q < -1 else int(q))
        zeros += t == 0
    return round(1.0 - zeros / len(xs), 3)
def flip(name):
    if name in LAND:
        side = "land"
    elif name in SEA:
        side = "sea"
    else:
        return {"ok": False, "why": "crossed", "model_pull": False}
    e = energy(name)
    band = "fail" if e < 0.40 else ("dense" if e > 0.85 else "pass")
    return {"ok": band == "pass", "engine": "koirig", "protocol": PROTOCOL,
            "side": side, "name": name, "energy": e, "band": band,
            "write": False, "model_pull": False}
