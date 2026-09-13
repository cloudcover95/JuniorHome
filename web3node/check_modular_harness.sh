#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
python3 - <<'PY'
import json, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path("web3node") if Path("web3node").is_dir() else Path(".")))
from sparse_formats import compare_formats
from stocks_via_llm import infer
import numpy as np
info = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()}
try:
    import mlx.core as mx
    info["mlx"] = True
except Exception:
    info["mlx"] = False
rng = np.random.default_rng(4)
info["dense_gauss"] = compare_formats(rng.normal(size=(48, 48)))
info["stocks"] = infer("q_mark field ticker")
print(json.dumps(info, indent=2, default=str))
PY
echo "[+] modular harness ok on $(uname -s)/$(uname -m)"
