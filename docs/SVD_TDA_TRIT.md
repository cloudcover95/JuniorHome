# Why svd_metal / tda_mesh were not rewritten

Those two files `import mlx` and `gtda` at module top. On T0 Linux that import dies.
Modern path is `svd_trit.py` + `tda_trit.py`. Same jobs. Numpy clock. Trit residual.
Metal/giotto remain the optional extra when the box has them.
