"""Economy / energy / market shift metrics. trit-pack + JSONL, not parquet."""
from __future__ import annotations
from typing import Any
import numpy as np
from macro_solver import as_returns, load_matrix, solve
from ph_algorithms import h0_persistence
from sparse_formats import compare_formats

def sector_split(returns: np.ndarray) -> dict[str, Any]:
    half = max(1, returns.shape[1] // 2)
    energy, macro = returns[:, :half], returns[:, half:]
    return {
        "energy_vol": float(np.std(energy)),
        "macro_vol": float(np.std(macro)),
        "energy_h0": h0_persistence(energy[: min(20, energy.shape[0])]),
        "macro_h0": h0_persistence(macro[: min(20, macro.shape[0])]),
    }

def metrics() -> dict[str, Any]:
    x = load_matrix()
    base = solve(x)
    split = sector_split(x if float(np.mean(np.abs(x))) < 2 else as_returns(x))
    return {
        "port": base["port"],
        "shape": base["shape"],
        "gamma": base["gamma"],
        "absmean_sparsity": base["absmean_sparsity"],
        "svd_k": base["tick"]["k"],
        "retained_energy": base["tick"]["retained_energy"],
        "vr": base["vr"],
        "sparse": compare_formats(x),
        "energy_vol": split["energy_vol"],
        "macro_vol": split["macro_vol"],
        "energy_h0_mean_death": split["energy_h0"]["mean_death"],
        "macro_h0_mean_death": split["macro_h0"]["mean_death"],
        "shift_ratio": split["macro_vol"] / (split["energy_vol"] + 1e-12),
    }

if __name__ == "__main__":
    import json
    print(json.dumps(metrics(), indent=2, default=str))
