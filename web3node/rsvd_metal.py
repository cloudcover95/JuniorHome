# web3node/rsvd_metal.py

import mlx.core as mx

def compute_hardware_rsvd(A: mx.array, target_rank: int, n_oversamples: int = 10, n_iter: int = 2) -> tuple:
    """
    Metal-accelerated Randomized Singular Value Decomposition (RSVD).
    Reduces O(N^3) bottleneck to O(N * rank^2) for interstellar sparse arrays.
    
    Mathematical Core:
    1. Gaussian random projection: $\Omega \sim \mathcal{N}(0, 1)$
    2. Sample column space: $Y = (A A^T)^q A \Omega$
    3. QR Factorization: $Y = Q R$
    4. Project and decompose: $B = Q^T A \rightarrow B = U_B \Sigma V^T \rightarrow U = Q U_B$
    """
    m, n = A.shape
    r = min(target_rank + n_oversamples, n)
    
    # 1. Generate Gaussian test matrix on MLX stream
    Omega = mx.random.normal((n, r), dtype=A.dtype)
    
    # 2. Subspace iteration (power method to filter high-frequency noise)
    Y = mx.matmul(A, Omega)
    for _ in range(n_iter):
        Y = mx.matmul(A, mx.matmul(A.T, Y))
        
    # 3. Orthogonalize via QR (MLX cpu stream fallback for stability if GPU QR is saturated)
    Q, _ = mx.linalg.qr(Y, stream=mx.cpu)
    
    # 4. Project A into the low-dimensional subspace
    B = mx.matmul(Q.T, A)
    
    # 5. Exact SVD on the small B matrix
    Ub, S, Vt = mx.linalg.svd(B, stream=mx.cpu)
    
    # 6. Reconstruct full U
    U = mx.matmul(Q, Ub)
    
    # Truncate to target rank
    return U[:, :target_rank], S[:target_rank], Vt[:target_rank, :]
