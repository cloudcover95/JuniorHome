import struct
from junior_gamma import quant_j
from music_pattern import HAT, KICK, SNAR, steps
from trit5 import pack5, unpack5
def pack(bpm=120):
    hits = steps(KICK, 0.9) + steps(SNAR, 0.75) + steps(HAT, 0.4)
    q, g = quant_j(hits)
    body = KICK.encode() + SNAR.encode() + HAT.encode()
    return b"JDB1" + struct.pack("<Hf", bpm, g) + body + pack5(q)
def unpack(blob):
    bpm, g = struct.unpack_from("<Hf", blob, 4)
    rows = blob[10:58]
    return {"magic": blob[:4].decode(), "bpm": bpm, "gamma_j": g,
            "kick": rows[:16].decode(), "snare": rows[16:32].decode(),
            "hat": rows[32:48].decode(), "n": len(unpack5(blob[58:])), "bytes": len(blob)}
