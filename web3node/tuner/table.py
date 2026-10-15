"""Local VE / ignition / AFR table. Trit cells. No ECU write."""
from __future__ import annotations
import json
from typing import Any
import numpy as np

class Table:
    def __init__(self, name: str, x: list[float], y: list[float], cells: list[list[float]]):
        self.name = name
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.cells = np.asarray(cells, dtype=np.float64)

    def scale(self, k: float) -> None:
        self.cells *= float(k)

    def smooth(self) -> None:
        c = self.cells
        if c.shape[0] < 3 or c.shape[1] < 3:
            return
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        out = c.copy()
        for i in range(1, c.shape[0] - 1):
            for j in range(1, c.shape[1] - 1):
                out[i, j] = np.sum(c[i-1:i+2, j-1:j+2] * k)
        self.cells = out

    def trit(self) -> np.ndarray:
        g = float(np.mean(np.abs(self.cells))) or 1.0
        return np.clip(np.rint(self.cells / g), -1, 1).astype(np.int8)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "x": self.x.tolist(), "y": self.y.tolist(),
                "cells": self.cells.tolist(), "trit": self.trit().tolist()}

if __name__ == "__main__":
    t = Table("ve", [1000, 2000, 3000], [0.2, 0.5, 0.8], [[0.8, 0.9, 1.0], [0.85, 0.95, 1.05], [0.9, 1.0, 1.1]])
    t.scale(1.02)
    print(json.dumps(t.to_json(), indent=2))
