#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VARIANTS="${VARIANTS:-meansparse extrasparse}"
exec "${SCRIPT_DIR}/submit_search_voc_train_all_sparse_cases.sh"
