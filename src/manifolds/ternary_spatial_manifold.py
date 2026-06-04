# path: src/manifolds/ternary_spatial_manifold.py

from dataclasses import dataclass
from typing import Optional, Dict, Any
import mlx.core as mx


@dataclass
class ManifoldConfig:
    dimension: int
    physics_informed: bool = False
    persistence_enabled: bool = True
    backend: str = "mlx"


class TernarySpatialManifold:
    """
    Core Ternary Spatial Manifold for JuniorCloud LLC ecosystem.

    Projects high-dimensional spatial/state data into the discrete {-1, 0, +1} domain
    while preserving topological structure. Supports physics-informed folding
    and long-running state across black boxes.
    """

    def __init__(self, config: ManifoldConfig):
        self.config = config
        self.state: Optional[mx.array] = None
        self.active_adapter_id: Optional[str] = None
        self.persistence: Dict[str, Any] = {} if config.persistence_enabled else None

    def project(self, data: mx.array) -> mx.array:
        scaled = data / (mx.mean(mx.abs(data)) + 1e-8)
        ternary = mx.where(scaled > 0.5, 1.0,
                  mx.where(scaled < -0.5, -1.0, 0.0))
        self.state = ternary.astype(mx.int8)
        return self.state

    def update(self, physics_prior: Optional[mx.array] = None, 
               damping: float = 0.1, energy_bound: float = 1.0) -> mx.array:
        if self.state is None:
            raise ValueError("No state to update")

        if self.config.physics_informed and physics_prior is not None:
            delta = physics_prior * 0.1
            self.state = mx.clip(self.state + delta, -energy_bound, energy_bound)

        self.state = self.state * (1.0 - damping)
        self.state = mx.clip(self.state, -1.0, 1.0).astype(mx.int8)
        return self.state

    def get_topological_features(self) -> Dict[str, float]:
        if self.state is None:
            return {}
        return {
            "mean_abs": float(mx.mean(mx.abs(self.state))),
            "sparsity": float(mx.mean(self.state == 0)),
        }
