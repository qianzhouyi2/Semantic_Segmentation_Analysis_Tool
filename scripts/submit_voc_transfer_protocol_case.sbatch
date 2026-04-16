#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_trproto_case
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
FEISHU_CONFIG="${HOME}/.feishu/config.json"
FEISHU_BRIDGE="${FEISHU_BRIDGE:-${HOME}/tools/feishu/interactive_bridge.py}"

CASE_JSON="${CASE_JSON:?CASE_JSON is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
EPSILON_SCALE="${EPSILON_SCALE:-1.0}"
EPSILON_RADIUS_255="${EPSILON_RADIUS_255:-}"
ATTACK_BACKWARD_MODE="${ATTACK_BACKWARD_MODE:-default}"
NUM_RESTARTS="${NUM_RESTARTS:-1}"
EOT_ITERS="${EOT_ITERS:-1}"

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
    feishu_notify "${SLURM_JOB_ID:-nojob}:fail transfer-protocol case rc=${rc}"
  fi
  exit $rc
}
trap on_exit EXIT

activate_segtool
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

case_id="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["case_id"])' "${CASE_JSON}")"
feishu_notify "${SLURM_JOB_ID:-nojob}:start transfer-protocol ${case_id}"

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  feishu_notify "${SLURM_JOB_ID:-nojob}:skip transfer-protocol ${case_id}"
  status_flag="done"
  exit 0
fi

attack_protocol_args=(
  --epsilon-scale "${EPSILON_SCALE}"
  --attack-backward-mode "${ATTACK_BACKWARD_MODE}"
  --num-restarts "${NUM_RESTARTS}"
  --eot-iters "${EOT_ITERS}"
)
if [[ -n "${EPSILON_RADIUS_255}" ]]; then
  attack_protocol_args+=(--epsilon-radius-255 "${EPSILON_RADIUS_255}")
fi

python scripts/run_transfer_group_attack.py \
  --case-json "${CASE_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${attack_protocol_args[@]}"

feishu_notify "${SLURM_JOB_ID:-nojob}:done transfer-protocol ${case_id}"
status_flag="done"
