LAYERS = {
    "T4_ship": {"where": "SRAM-class", "keep": "trit5 payload", "drop": "GGUF header"},
    "T0_m6": {"where": "unified 24GB", "keep": "Trit3 + JTR1", "drop": "UE boot"},
    "T1_spark": {"where": "128GB", "keep": "optional Q4_K-shaped", "drop": "not required for ticket"},
}
def map():
    return {"layout": "tag,rows,gamma,cols,trit5", "align": 1, "layers": LAYERS}
def stack():
    return map()
