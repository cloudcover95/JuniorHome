#!/usr/bin/env zsh
# Additive gate for web3node/ternary_tda.py
# Does not docker, does not touch main, does not write /opt or /vault.
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
python3 web3node/ternary_tda.py
if python3 -c "import mlx.core as mx; print(mx.__version__)" >/dev/null 2>&1; then
  echo "[+] mlx present — Metal path available on this box"
else
  echo "[*] mlx absent — numpy fallback used (ok on this host)"
fi
echo "[+] ternary_tda tick passed"
