from music_smf import smf
JUNIOR = {"kick": 36, "snare": 38, "hat": 42}
GM = {"kick": 36, "snare": 38, "hat": 42, "open_hat": 46, "clap": 39, "tom": 45}
PAD16 = {f"p{i}": 36 + i for i in range(16)}
def apply(name="junior"):
    maps = {"junior": JUNIOR, "gm": GM, "pad16": PAD16}
    chosen = maps.get(name, JUNIOR)
    used = {k: chosen.get(k, JUNIOR[k]) for k in JUNIOR}
    mid = smf(notes=used)
    return {"map": name, "notes": chosen, "smf_B": mid["bytes"], "compat": ("smf0", "note-on", "ch0")}
