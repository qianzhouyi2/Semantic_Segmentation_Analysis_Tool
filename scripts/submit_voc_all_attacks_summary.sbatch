#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_allatt_sum
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
FEISHU_CONFIG="${HOME}/.feishu/config.json"
FEISHU_BRIDGE="${FEISHU_BRIDGE:-${HOME}/tools/feishu/interactive_bridge.py}"

MANIFEST="${MANIFEST:?MANIFEST is required}"
SUITE_ROOT="${SUITE_ROOT:?SUITE_ROOT is required}"
EXCLUDE_ATTACK_STEMS="${EXCLUDE_ATTACK_STEMS:-}"
PER_SAMPLE_POLICY="${PER_SAMPLE_POLICY:-auto}"

feishu_notify() {
  local text="$1"
  if [[ -f "${FEISHU_CONFIG}" && -f "${FEISHU_BRIDGE}" ]]; then
    python "${FEISHU_BRIDGE}" --config "${FEISHU_CONFIG}" send-text "$text" >/dev/null 2>&1 || true
  fi
}

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

status_flag="running"
on_exit() {
  local rc=$?
  if [[ "${status_flag}" == "running" && $rc -ne 0 ]]; then
    feishu_notify "${SLURM_JOB_ID:-nojob}:fail all-attacks summary rc=${rc}"
  fi
  exit $rc
}
trap on_exit EXIT

activate_segtool
cd "${PROJECT_ROOT}"

feishu_notify "${SLURM_JOB_ID:-nojob}:start all-attacks summary"
exclude_args=()
if [[ -n "${EXCLUDE_ATTACK_STEMS}" ]]; then
  exclude_args+=(--exclude-attack-stems "${EXCLUDE_ATTACK_STEMS}")
fi
python scripts/summarize_voc_all_attacks.py \
  --manifest "${MANIFEST}" \
  --suite-root "${SUITE_ROOT}" \
  --per-sample-policy "${PER_SAMPLE_POLICY}" \
  "${exclude_args[@]}"
feishu_notify "${SLURM_JOB_ID:-nojob}:done all-attacks summary outputs=${SUITE_ROOT}/summary"
status_flag="done"
