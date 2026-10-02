"""RP2040 mux for the Geode deck. Conflict check. Not flashed."""
PINS = {"key_row": list(range(0, 8)), "key_col": list(range(8, 16)),
        "pad_row": list(range(16, 20)), "pad_col": list(range(20, 24)),
        "gain": [26], "filter": [27], "tempo": [28], "mix": [29]}
ADC = {26, 27, 28, 29}
def mux():
    used = {}
    clash = []
    for name, pins in PINS.items():
        for pin in pins:
            if pin in used:
                clash.append([pin, used[pin], name])
            used[pin] = name
    return {"mcu": "rp2040-soft", "gpio_used": len(used), "adc": sorted(ADC & set(used)),
            "display": "dsi", "clash": clash, "ok": not clash and len(used) <= 30,
            "trit_mcu": False, "flashed": False, "live": False}
