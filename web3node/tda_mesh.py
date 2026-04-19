# web3node/tda_mesh.py

import mlx.core as mx
import numpy as np
from gtda.homology import VietorisRipsPersistence

def compute_persistent_homology(tensor_state: mx.array, max_homology_dim: int = 1) -> mx.array:
    """
    Extracts topological invariants (Betti numbers) via Vietoris-Rips complexes.
    Maps $N$-dimensional financial/market noise into persistence diagrams.
    """
    # 1. Cast MLX unified memory tensor to dense Numpy array for giotto-tda
    point_cloud = np.array(tensor_state.tolist(), dtype=np.float32)
    
    # 2. giotto-tda requires shape: (n_samples, n_points, n_dimensions)
    if point_cloud.ndim == 2:
        point_cloud = point_cloud[None, :, :]
        
    # 3. Compute Vietoris-Rips Persistence
    vr = VietorisRipsPersistence(homology_dimensions=list(range(max_homology_dim + 1)))
    persistence_diagrams = vr.fit_transform(point_cloud)
    
    # 4. Re-cast the topological features back to MLX for continuous pipeline execution
    return mx.array(persistence_diagrams.tolist())
