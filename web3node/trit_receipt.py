"""Trit receipt. Not ML-DSA. Not quantum-safe. Not a chain signature."""
import hashlib
def pack(text):
    xs = [((ord(c) % 5) - 2) / 2.0 for c in (text or "x")[:32]]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in trits)
def receipt(address, text, secret):
    body = pack(text)
    mac = hashlib.sha256(secret + address.encode() + body).hexdigest()[:32]
    return {"scheme": "trit-receipt", "address": address, "n": len(body), "mac": mac,
            "pq_safe": False, "ml_dsa": False, "chain": False}
def verify(row, text, secret):
    return receipt(row["address"], text, secret)["mac"] == row["mac"]
