#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=sparse_thr_case
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
FAMILY="${FAMILY:?FAMILY is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
VARIANT="${VARIANT:?VARIANT is required}"

DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
ATTACK_CONFIG="${ATTACK_CONFIG:-${PROJECT_ROOT}/configs/attacks/pgd.yaml}"
THRESHOLDS="${THRESHOLDS:-0.05 0.10 0.15 0.20 0.25 0.35 0.40}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_threshold_search}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${PROJECT_ROOT}/models/defenses}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE_CLEAN="${BATCH_SIZE_CLEAN:-8}"
BATCH_SIZE_ADV="${BATCH_SIZE_ADV:-2}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-4}"
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

ensure_sidecar() {
  local checkpoint_stem
  checkpoint_stem="$(basename "${CHECKPOINT}" .pth)"
  local output_dir="${SIDECAR_ROOT}/${VARIANT}"
  local stats_path="${output_dir}/${checkpoint_stem}_${VARIANT}_${DATASET_SPLIT}.pt"
  local summary_path="${output_dir}/${checkpoint_stem}_${VARIANT}_${DATASET_SPLIT}.summary.json"
  local config_path="${PROJECT_ROOT}/configs/defenses/${VARIANT}_example.yaml"
  mkdir -p "${output_dir}"
  if [[ ! -f "${stats_path}" ]]; then
    log "Preparing sidecar variant=${VARIANT} family=${FAMILY} checkpoint=${CHECKPOINT}" >&2
    python "${PROJECT_ROOT}/scripts/prepare_sparse_defense.py" \
      --family "${FAMILY}" \
      --checkpoint "${PROJECT_ROOT}/${CHECKPOINT}" \
      --defense-config "${config_path}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-split "${DATASET_SPLIT}" \
      --output-stats "${stats_path}" \
      --summary-path "${summary_path}" \
      --batch-size "${CALIB_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}"
  fi
  printf '%s\n' "${stats_path}"
}

activate_segtool
cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}"

CHECKPOINT_STEM="$(basename "${CHECKPOINT}" .pth)"
OUTPUT_DIR="${RESULTS_ROOT}/${CHECKPOINT_STEM}/${VARIANT}"
STATS_PATH="$(ensure_sidecar)"

log "Starting threshold search case"
log "family=${FAMILY}"
log "checkpoint=${CHECKPOINT}"
log "variant=${VARIANT}"
log "stats_path=${STATS_PATH}"
log "dataset_root=${DATASET_ROOT}"
log "dataset_split=${DATASET_SPLIT}"
log "thresholds=${THRESHOLDS}"

python "${PROJECT_ROOT}/scripts/search_sparse_thresholds.py" \
  --family "${FAMILY}" \
  --checkpoint "${PROJECT_ROOT}/${CHECKPOINT}" \
  --variant "${VARIANT}" \
  --stats-path "${STATS_PATH}" \
  --attack-config "${ATTACK_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-split "${DATASET_SPLIT}" \
  --thresholds "${THRESHOLDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size-clean "${BATCH_SIZE_CLEAN}" \
  --batch-size-adv "${BATCH_SIZE_ADV}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

log "Finished threshold search case output_dir=${OUTPUT_DIR}"
