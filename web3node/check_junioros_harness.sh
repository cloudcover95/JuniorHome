#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 junioros_harness.py
test -f vault/junioros_harness.json
echo "[junioros] harness ok"
