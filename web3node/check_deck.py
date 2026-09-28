from beat_pack import pack, unpack
from music_deck import deck
from music_pattern import pattern
from music_wire import wire
def check():
    u = unpack(pack())
    return {"probe": deck()["apps"], "live": False, "wire_n": len(wire()["graph"]),
            "pack_B": u["bytes"], "pattern_hits": pattern()["pad"]["hits"],
            "ok": u["magic"]=="JDB1" and u["bytes"]==70}
