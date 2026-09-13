#!/usr/bin/env zsh
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
bash web3node/check_modular_harness.sh "$ROOT"
