"""JuniorDeck engine. Step ticket stays in RAM until the envelope closes."""
import time
from pathlib import Path
from write_gate import commit
OUT = Path.home() / ".juniorhome" / "deck" / "engine.txt"
def steps(n=16):
    return [1.0 if i % 4 == 0 else (0.4 if i % 2 == 0 else 0.0) for i in range(n)]
def pack(xs):
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return f"n={len(trits)} z={trits.count(0)} g={gamma:.3f}"
def run(env="open"):
    t0 = time.perf_counter()
    text = pack(steps())
    row = commit(env, OUT, text)
    return {"env": env, "ticket": text, "us": round((time.perf_counter() - t0) * 1e6, 1),
            "ram": not row["disk"], "disk": row["disk"], "why": row["why"],
            "ardour": False, "live": False, "model_pull": False}
