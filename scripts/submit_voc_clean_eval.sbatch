#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_clean_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/home/qianzhouyi/projects/Semantic_Segmentation_Analysis_Tool"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_clean_eval}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
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

mkdir -p "${RESULTS_ROOT}"
activate_segtool
cd "${PROJECT_ROOT}"

log "Starting VOC clean evaluation on slurm"
log "project_root=${PROJECT_ROOT}"
log "results_root=${RESULTS_ROOT}"
log "dataset_root=${DATASET_ROOT}"
log "batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} device=${DEVICE}"

python scripts/evaluate_all_voc_clean_models.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${RESULTS_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

log "Finished VOC clean evaluation"
