from ste_tape import step
def run(steps=8):
    w, x = [0.3, -0.2, 0.1], [1.0, 0.5, -0.2]
    hist = []
    for _ in range(steps):
        row = step(w, x, 0.0, lr=0.08)
        w = row["w"]; hist.append(row["err"])
    return {"steps": steps, "err0": hist[0], "errN": hist[-1], "dropped": hist[-1] < hist[0], "ste": True}
