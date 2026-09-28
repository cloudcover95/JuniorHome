from music_clock import clock
from music_deck import pad
KICK, SNAR, HAT = "1000100010001000", "0000100000001000", "1010101010101010"
def steps(row, vel=0.8):
    return [vel if c == "1" else 0.0 for c in row]
def pattern():
    hits = steps(KICK, 0.9) + steps(SNAR, 0.75) + steps(HAT, 0.4)
    row = clock()
    row["bars"] = 1
    row["pattern"] = {"kick": KICK, "snare": SNAR, "hat": HAT}
    row["pad"] = pad(hits)
    return row
