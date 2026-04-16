#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_eval_case
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FEISHU_CONFIG="${HOME}/.feishu/config.json"
FEISHU_BRIDGE="${FEISHU_BRIDGE:-${HOME}/tools/feishu/interactive_bridge.py}"

MODE="${MODE:?MODE is required}"  # clean | attack
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
FAMILY="${FAMILY:?FAMILY is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

DEFENSE_CONFIG="${DEFENSE_CONFIG:-}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
ATTACK_CONFIG="${ATTACK_CONFIG:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

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
    feishu_notify "${SLURM_JOB_ID:-nojob}:fail ${MODE} ${MODEL_ID} rc=${rc}"
  fi
  exit $rc
}
trap on_exit EXIT

activate_segtool
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

feishu_notify "${SLURM_JOB_ID:-nojob}:start ${MODE} ${MODEL_ID}"

defense_args=()
if [[ -n "${DEFENSE_CONFIG}" ]]; then
  defense_args+=(--defense-config "${DEFENSE_CONFIG}")
fi

if [[ "${MODE}" == "clean" ]]; then
  if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    feishu_notify "${SLURM_JOB_ID:-nojob}:skip clean ${MODEL_ID}"
    status_flag="done"
    exit 0
  fi
  python scripts/evaluate_voc_clean.py \
    --family "${FAMILY}" \
    --checkpoint "${CHECKPOINT}" \
    "${defense_args[@]}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
elif [[ "${MODE}" == "attack" ]]; then
  if [[ -z "${ATTACK_CONFIG}" ]]; then
    echo "ATTACK_CONFIG is required for MODE=attack" >&2
    exit 1
  fi
  if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    feishu_notify "${SLURM_JOB_ID:-nojob}:skip attack ${MODEL_ID} $(basename "${ATTACK_CONFIG}" .yaml)"
    status_flag="done"
    exit 0
  fi
  python scripts/run_attack.py \
    --attack-config "${ATTACK_CONFIG}" \
    --family "${FAMILY}" \
    --checkpoint "${CHECKPOINT}" \
    "${defense_args[@]}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
else
  echo "Unsupported MODE=${MODE}" >&2
  exit 1
fi

feishu_notify "${SLURM_JOB_ID:-nojob}:done ${MODE} ${MODEL_ID} output=${OUTPUT_DIR}"
status_flag="done"
