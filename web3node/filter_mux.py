"""AINSEL is one channel. Filter takes it and clears the others."""
HOLD = {"ainsel": None}
def select_filter():
    HOLD["ainsel"] = 1
    return {"ok": True, "control": "filter", "funcsel": "null", "ainsel": 1,
            "pins": [27], "others": False, "input_enabled": False, "programmed": False}
