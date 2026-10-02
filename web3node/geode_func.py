"""FUNCSEL and AINSEL. Filter holds AINSEL alone."""
MATRIX = set(range(24))
ADC = {26: 0, 27: 1, 28: 2, 29: 3}
CONTROLS = {"note": ("funcsel", list(range(16))), "step": ("funcsel", list(range(16, 24))),
            "gamma": ("ainsel", [26]), "filter": ("ainsel", [27]),
            "clock": ("ainsel", [28]), "mix": ("ainsel", [29])}
HOLD = {"ainsel": None}
def select(name):
    row = CONTROLS.get(name)
    if row is None:
        return {"ok": False, "why": "unknown", "programmed": False}
    kind, pins = row
    if kind == "ainsel":
        HOLD["ainsel"] = ADC[pins[0]]
        return {"ok": True, "control": name, "funcsel": "null", "ainsel": HOLD["ainsel"],
                "others": False, "input_enabled": False, "pins": pins, "programmed": False}
    if any(p in ADC for p in pins):
        return {"ok": False, "why": "adc_pin", "programmed": False}
    return {"ok": True, "control": name, "funcsel": "sio", "ainsel": None,
            "input_enabled": True, "pins": pins, "programmed": False}
