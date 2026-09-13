"""Lean sparse formats. No scipy. COO / CSR / CSC / trit-CSR."""
from __future__ import annotations
from typing import Any
import numpy as np
from bitnet_orig import absmean
from trit_cache import pack_trits

def to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "tolist"):
        return np.array(matrix.tolist(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)

def coo(matrix: Any) -> dict[str, Any]:
    dense = to_dense(matrix)
    rows, cols = np.nonzero(dense)
    data = dense[rows, cols]
    return {"fmt": "coo", "shape": dense.shape, "nnz": int(data.size),
            "row": rows.astype(np.int32), "col": cols.astype(np.int32),
            "data": data.astype(np.float32)}

def csr(matrix: Any) -> dict[str, Any]:
    dense = to_dense(matrix)
    rows, cols = np.nonzero(dense)
    data = dense[rows, cols].astype(np.float32)
    counts = np.bincount(rows, minlength=dense.shape[0]).astype(np.int32)
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)
    return {"fmt": "csr", "shape": dense.shape, "nnz": int(data.size),
            "data": data, "indices": cols.astype(np.int32), "indptr": indptr}

def csc(matrix: Any) -> dict[str, Any]:
    dense = to_dense(matrix)
    rows, cols = np.nonzero(dense)
    order = np.argsort(cols, kind="stable")
    rows, cols = rows[order], cols[order]
    data = dense[rows, cols].astype(np.float32)
    counts = np.bincount(cols, minlength=dense.shape[1]).astype(np.int32)
    indptr = np.zeros(dense.shape[1] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)
    return {"fmt": "csc", "shape": dense.shape, "nnz": int(data.size),
            "data": data, "indices": rows.astype(np.int32), "indptr": indptr}

def trit_csr(matrix: Any) -> dict[str, Any]:
    dense = to_dense(matrix)
    flat, scale = absmean(dense.reshape(-1).tolist())
    q = np.asarray(flat, dtype=np.int8).reshape(dense.shape)
    packed = csr(q)
    packed["fmt"] = "trit-csr"
    packed["scale"] = scale
    packed["data"] = packed["data"].astype(np.int8)
    packed["sparsity"] = float((q == 0).mean())
    return packed

def footprint(pack: dict[str, Any]) -> int:
    total = 16
    for value in pack.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total

def compare_formats(matrix: Any) -> dict[str, Any]:
    dense = to_dense(matrix)
    packs = {
        "float32": int(dense.astype(np.float32).nbytes),
        "coo": footprint(coo(dense)),
        "csr": footprint(csr(dense)),
        "csc": footprint(csc(dense)),
        "trit_csr": footprint(trit_csr(dense)),
        "trit_pack": len(pack_trits(np.clip(np.rint(dense / (np.mean(np.abs(dense)) + 1e-7)), -1, 1))),
    }
    return {"bytes": packs, "winner": min(packs, key=packs.get),
            "shape": list(dense.shape), "nnz": int(np.count_nonzero(dense))}
