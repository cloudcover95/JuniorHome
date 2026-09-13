"""Home-node compose. FrameForge stack if importable, else pad-sign trit.

Stdlib. BitNet / TritARM score intent. Sim owns knockback.
Not a Nintendo product.
"""

from __future__ import annotations


def eval_features(feat):
    import importlib

    for name in ("python.frameforge.trit_stack", "frameforge.trit_stack"):
        try:
            mod = importlib.import_module(name)
            return mod.stack_eval(list(feat))
        except Exception:
            continue
    lx = feat[0] if feat else 0.0
    ly = feat[1] if len(feat) > 1 else 0.0
    trit = 1 if lx > 0.2 else (-1 if lx < -0.2 else 0)
    label = "right" if trit > 0 else ("left" if trit < 0 else ("jump" if ly > 0.2 else "wait"))
    return {
        "bitnet": {"action": label},
        "arm": {"trit": trit, "label": label},
        "envelope": {
            "port": "JuniorBitNetFieldCore",
            "trit": trit,
            "note": "fallback; install FrameForge python path",
        },
        "authority": "python_sim",
    }
