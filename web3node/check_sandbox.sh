#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 sandbox_suite.py
test -f vault/sandbox_suite.json
echo "[sandbox] numpy clock suite ok"
