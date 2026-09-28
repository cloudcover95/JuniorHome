import shutil
from junior_gamma import quant_j
from trit5 import pack5
APPS = {"ardour": ("ardour", "ardour8", "Ardour"), "hydrogen": ("hydrogen",), "audacity": ("audacity",)}
def probe():
    return {n: next((b for b in bins if shutil.which(b)), None) for n, bins in APPS.items()}
def pad(hits):
    q, g = quant_j(hits or [0.0])
    return {"hits": len(hits), "gamma_j": g, "trit5": len(pack5(q)), "q_head": q[:8]}
def deck():
    return {"apps": probe(), "live": False, "jack": False,
            "daw": "host binary, not in-tree",
            "pad": pad([0.2, 0.8, 0.1, 0.9, 0.0, 0.4]),
            "t4": "MIDI trit ticket only", "t0": "Ardour+Hydrogen on JACK/PipeWire"}
