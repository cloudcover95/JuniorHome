"""FUNCSEL and AINSEL for deck controls. Not programmed on this host."""
FUNCSEL = {"null": 31, "sio": 5, "pio0": 6, "pio1": 7}
AINSEL = {"gain": 0, "filter": 1, "tempo": 2, "mix": 3}
CONTROLS = {
    "note": {"mux": "funcsel", "value": "sio", "pins": "0-15"},
    "step": {"mux": "funcsel", "value": "sio", "pins": "16-23"},
    "gamma": {"mux": "ainsel", "value": 0, "pins": "26"},
    "filter": {"mux": "ainsel", "value": 1, "pins": "27"},
    "clock": {"mux": "ainsel", "value": 2, "pins": "28"},
    "mix": {"mux": "ainsel", "value": 3, "pins": "29"},
}
def select(name):
    row = CONTROLS.get(name)
    if row is None:
        return {"ok": False, "why": "unknown", "programmed": False}
    return {"ok": True, "control": name, "mux": row["mux"], "value": row["value"],
            "pins": row["pins"], "programmed": False, "model_pull": False}
