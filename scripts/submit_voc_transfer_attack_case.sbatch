#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_transfer_case
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

ATTACK_CONFIG="${ATTACK_CONFIG:?ATTACK_CONFIG is required}"
SOURCE_MODEL_ID="${SOURCE_MODEL_ID:?SOURCE_MODEL_ID is required}"
SOURCE_FAMILY="${SOURCE_FAMILY:?SOURCE_FAMILY is required}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:?TARGET_MODEL_ID is required}"
TARGET_FAMILY="${TARGET_FAMILY:?TARGET_FAMILY is required}"
TARGET_CHECKPOINT="${TARGET_CHECKPOINT:?TARGET_CHECKPOINT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

SOURCE_DEFENSE_CONFIG="${SOURCE_DEFENSE_CONFIG:-}"
TARGET_DEFENSE_CONFIG="${TARGET_DEFENSE_CONFIG:-}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
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
    feishu_notify "${SLURM_JOB_ID:-nojob}:fail transfer ${TARGET_MODEL_ID} rc=${rc}"
  fi
  exit $rc
}
trap on_exit EXIT

activate_segtool
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

attack_name="$(basename "${ATTACK_CONFIG}" .yaml)"
feishu_notify "${SLURM_JOB_ID:-nojob}:start transfer ${attack_name} ${SOURCE_MODEL_ID}->${TARGET_MODEL_ID}"

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  feishu_notify "${SLURM_JOB_ID:-nojob}:skip transfer ${attack_name} ${TARGET_MODEL_ID}"
  status_flag="done"
  exit 0
fi

source_defense_args=()
target_defense_args=()
if [[ -n "${SOURCE_DEFENSE_CONFIG}" ]]; then
  source_defense_args+=(--source-defense-config "${SOURCE_DEFENSE_CONFIG}")
fi
if [[ -n "${TARGET_DEFENSE_CONFIG}" ]]; then
  target_defense_args+=(--target-defense-config "${TARGET_DEFENSE_CONFIG}")
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

python scripts/run_transfer_attack.py \
  --attack-config "${ATTACK_CONFIG}" \
  --source-family "${SOURCE_FAMILY}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  "${source_defense_args[@]}" \
  --target-family "${TARGET_FAMILY}" \
  --target-checkpoint "${TARGET_CHECKPOINT}" \
  "${target_defense_args[@]}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${attack_protocol_args[@]}"

feishu_notify "${SLURM_JOB_ID:-nojob}:done transfer ${attack_name} ${TARGET_MODEL_ID} output=${OUTPUT_DIR}"
status_flag="done"
