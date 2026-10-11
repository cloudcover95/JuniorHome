"""Node box. Key stays here. No peers. No power reading."""
from dilithium_py.ml_dsa import ML_DSA_65
def box(tenant="cu-aurora"):
    pk, sk = ML_DSA_65.keygen()
    del sk
    return {"tenant": tenant, "pk_len": len(pk), "key_exported": False, "cold": True,
            "sweep": False, "peers": 0, "power_w": None, "governed": False,
            "why": "no_peers", "scheme": "ML-DSA-65", "network": False}
