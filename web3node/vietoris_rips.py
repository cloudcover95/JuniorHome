"""Lean Vietoris-Rips 1-skeleton.

tda_mesh.py still owns the giotto-tda persistence path.
This module builds the VR graph without that dependency:

- pairwise distances
- edges at scale epsilon
- Betti_0 via union-find
- graph H1 proxy = E - V + C (1-skeleton cycle rank)

That is not a full simplicial H1. It is enough to bench filtration
cost next to SVD retention on the home node.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_distance(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.square(delta).sum(axis=-1))


def vr_edges(distance: np.ndarray, epsilon: float) -> list[tuple[int, int, float]]:
    n = distance.shape[0]
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance[i, j])
            if d <= epsilon:
                edges.append((i, j, d))
    return edges


def betti0(vertex_count: int, edges: list[tuple[int, int, float]]) -> int:
    parent = list(range(vertex_count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    components = vertex_count
    for i, j, _ in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            components -= 1
    return components


def vr_summary(points: Any, epsilon: float | None = None) -> dict[str, float | int]:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2:
        raise ValueError("points must be 2D (n, d)")
    dist = pairwise_distance(cloud)
    off = dist[np.triu_indices(cloud.shape[0], k=1)]
    if epsilon is None:
        epsilon = float(np.median(off)) if off.size else 0.0
    edges = vr_edges(dist, epsilon)
    b0 = betti0(cloud.shape[0], edges)
    e = len(edges)
    v = cloud.shape[0]
    return {
        "n": v,
        "dim": cloud.shape[1],
        "epsilon": float(epsilon),
        "edges": e,
        "betti0": b0,
        "h1_graph": int(e - v + b0),
    }
