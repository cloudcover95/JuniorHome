from music_prod import prod
def clock(bpm=120.0, ppq=24):
    row = prod()
    row["bpm"] = bpm
    row["ppq"] = ppq
    row["tick_s"] = 60.0 / (bpm * ppq)
    row["transport"] = "internal"
    return row
