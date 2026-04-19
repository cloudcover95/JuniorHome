# src/physiomanifold/tda_svd_core/manifold_collapse.py

import mlx.core as mx
from web3node.rsvd_metal import compute_hardware_rsvd

class ThermodynamicManifold:
    """
    Enforces the Free-Energy Principle and Thermodynamic Conservation.
    Filters market/physical noise by measuring the Shannon entropy of the singular value manifold.
    """
    def __init__(self, target_rank: int = 64):
        self.target_rank = target_rank

    def calculate_state_entropy(self, singular_values: mx.array) -> mx.array:
        """
        Calculates the normalized Von Neumann-inspired entropy of the manifold.
        $H = -\sum_{i} p_i \ln p_i$ where $p_i = \sigma_i / \sum \sigma_j$
        """
        # Normalize singular values to a probability distribution
        s_sum = mx.sum(singular_values)
        if s_sum == 0:
            return mx.array(0.0)
            
        p = singular_values / s_sum
        # Add epsilon to prevent log(0) NaN propagation
        epsilon = 1e-9
        entropy = -mx.sum(p * mx.log(p + epsilon))
        return entropy

    def collapse_and_measure(self, state_tensor: mx.array) -> tuple:
        """
        Executes RSVD and computes the physical entropy invariant.
        """
        U, S, Vt = compute_hardware_rsvd(state_tensor, target_rank=self.target_rank)
        entropy = self.calculate_state_entropy(S)
        
        # Reconstruct the denoised manifold matrix: $A_{approx} = U \Sigma V^T$
        denoised_manifold = mx.matmul(U, mx.matmul(mx.diag(S), Vt))
        
        return denoised_manifold, entropy
