"""H0 persistence via VR filtration + union-find. giotto stays in tda_mesh.py."""
from __future__ import annotations
from typing import Any
import numpy as np
from vietoris_rips import pairwise_distance

def h0_persistence(points: np.ndarray) -> dict[str, Any]:
    cloud = np.asarray(points, dtype=np.float64)
    n = cloud.shape[0]
    dist = pairwise_distance(cloud)
    edges = [(float(dist[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    edges.sort()
    parent = list(range(n))
    rank = [0] * n
    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pairs = []
    leftover = n
    for scale, i, j in edges:
        a, b = find(i), find(j)
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        pairs.append((0.0, scale))
        leftover -= 1
        if leftover == 1:
            break
    deaths = [d for _, d in pairs]
    return {
        "algorithm": "vietoris-rips H0 union-find",
        "n": n,
        "finite_pairs": len(pairs),
        "infinite_bars": leftover,
        "mean_death": float(np.mean(deaths)) if deaths else 0.0,
        "max_death": float(max(deaths) if deaths else 0.0),
        "pairs_head": pairs[:8],
    }

def named_algorithms() -> dict[str, str]:
    return {
        "Vietoris-Rips": "live lean + giotto tda_mesh",
        "H0 union-find": "live lean ph_algorithms.h0_persistence",
        "Ripser": "not vendored — optional extra",
        "Alpha / Cech": "named only",
        "Witness": "named only",
        "GUDHI": "named only",
    }
