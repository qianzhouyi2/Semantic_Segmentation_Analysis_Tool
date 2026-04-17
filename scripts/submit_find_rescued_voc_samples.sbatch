#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_rescued
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

FAMILY="${FAMILY:?FAMILY is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:?DEFENSE_CONFIG is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"

DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-val}"
ATTACK_CONFIG="${ATTACK_CONFIG:-}"
METRIC="${METRIC:-sample_miou}"
MIN_DELTA="${MIN_DELTA:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
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
    log "Unable to locate conda.sh"
    return 1
  fi
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate segtool
}

activate_segtool
cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" logs

log "Starting rescued VOC sample search"
log "family=${FAMILY}"
log "checkpoint=${CHECKPOINT}"
log "defense_config=${DEFENSE_CONFIG}"
log "dataset_root=${DATASET_ROOT}"
log "dataset_split=${DATASET_SPLIT}"
log "attack_config=${ATTACK_CONFIG:-<none>}"
log "metric=${METRIC}"
log "min_delta=${MIN_DELTA}"
log "batch_size=${BATCH_SIZE}"
log "num_workers=${NUM_WORKERS}"
log "device=${DEVICE}"
log "output_json=${OUTPUT_JSON}"

attack_args=()
if [[ -n "${ATTACK_CONFIG}" ]]; then
  attack_args+=(--attack-config "${ATTACK_CONFIG}")
fi

python scripts/find_rescued_voc_samples.py \
  --family "${FAMILY}" \
  --checkpoint "${CHECKPOINT}" \
  --defense-config "${DEFENSE_CONFIG}" \
  "${attack_args[@]}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-split "${DATASET_SPLIT}" \
  --metric "${METRIC}" \
  --min-delta "${MIN_DELTA}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --output-json "${OUTPUT_JSON}"

log "Finished rescued VOC sample search output_json=${OUTPUT_JSON}"
