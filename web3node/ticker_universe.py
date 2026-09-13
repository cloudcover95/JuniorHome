ENERGY = ("XOM", "CVX", "COP", "XLE", "USO")
MACRO = ("SPY", "TLT", "GLD", "UUP", "HYG")
UNIVERSE = ENERGY + MACRO
SECTOR = {name: "energy" for name in ENERGY} | {name: "macro" for name in MACRO}
