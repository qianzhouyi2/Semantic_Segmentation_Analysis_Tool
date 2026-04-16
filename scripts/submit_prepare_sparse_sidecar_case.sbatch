#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=sparse_sidecar_case
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
FAMILY="${FAMILY:?FAMILY is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
VARIANT="${VARIANT:?VARIANT is required}"

DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${PROJECT_ROOT}/models/defenses}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
DEFENSE_TEMPLATE_CONFIG="${DEFENSE_TEMPLATE_CONFIG:-${PROJECT_ROOT}/configs/defenses/${VARIANT}_example.yaml}"

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

checkpoint_stem="$(basename "${CHECKPOINT}" .pth)"
output_dir="${SIDECAR_ROOT}/${VARIANT}"
output_stats="${output_dir}/${checkpoint_stem}_${VARIANT}_${DATASET_SPLIT}.pt"
summary_path="${output_dir}/${checkpoint_stem}_${VARIANT}_${DATASET_SPLIT}.summary.json"

mkdir -p "${output_dir}"

log "Starting sparse sidecar prepare case"
log "family=${FAMILY}"
log "checkpoint=${CHECKPOINT}"
log "variant=${VARIANT}"
log "dataset_root=${DATASET_ROOT}"
log "dataset_split=${DATASET_SPLIT}"
log "device=${DEVICE}"
log "batch_size=${BATCH_SIZE}"
log "num_workers=${NUM_WORKERS}"
log "defense_template_config=${DEFENSE_TEMPLATE_CONFIG}"
log "output_stats=${output_stats}"
log "force_rebuild=${FORCE_REBUILD}"

if [[ "${FORCE_REBUILD}" != "1" && -f "${output_stats}" ]]; then
  log "Sidecar already exists, skipping prepare"
  exit 0
fi

python "${PROJECT_ROOT}/scripts/prepare_sparse_defense.py" \
  --family "${FAMILY}" \
  --checkpoint "${PROJECT_ROOT}/${CHECKPOINT}" \
  --defense-config "${DEFENSE_TEMPLATE_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-split "${DATASET_SPLIT}" \
  --output-stats "${output_stats}" \
  --summary-path "${summary_path}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

log "Finished sparse sidecar prepare case output=${output_stats}"
