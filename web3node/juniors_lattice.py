"""Juniors lattice note. Trit pack is the message, not the lattice."""
from dilithium_py.ml_dsa import ML_DSA_65
def lattice():
    return {"scheme": "ML-DSA-65", "family": "module-lattice", "k": ML_DSA_65.k,
            "l": ML_DSA_65.l, "eta": ML_DSA_65.eta, "trit_role": "message",
            "lattice_is_trit": False, "pq_ready": True, "network": False}
