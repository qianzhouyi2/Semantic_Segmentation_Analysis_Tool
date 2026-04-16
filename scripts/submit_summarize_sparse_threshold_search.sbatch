#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=sparse_thr_sum
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEARCH_ROOT="${SEARCH_ROOT:?SEARCH_ROOT is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${SEARCH_ROOT}}"

activate_segtool() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "segtool" ]]; then
    return 0
  fi
  local conda_base=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  elif command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
  elif [[ -d "/opt/miniconda3" ]]; then
    conda_base="/opt/miniconda3"
  fi
  if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    return 1
  fi
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate segtool
}

activate_segtool
cd "${PROJECT_ROOT}"
python "${PROJECT_ROOT}/scripts/summarize_sparse_threshold_search.py" \
  --search-root "${SEARCH_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
