#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_sparse_sidecars
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEFENSE_CONFIG_DIR="${DEFENSE_CONFIG_DIR:-${PROJECT_ROOT}/configs/defenses}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/models/defenses}"
VARIANTS="${VARIANTS:-all}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

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

export VARIANTS
mapfile -t VARIANT_ROWS < <(python - <<'PY'
import os

from src.common.sparse_workflow import parse_sparse_variants

for item in parse_sparse_variants(os.environ.get("VARIANTS"), default=("all",)):
    print(item)
PY
)

mapfile -t MODEL_ROWS < <(python - <<'PY'
from src.common.voc_protocol import VOC_BASE_MODELS

for item in VOC_BASE_MODELS:
    print(f"{item['family']} {item['checkpoint']}")
PY
)

mkdir -p "${OUTPUT_ROOT}"

log "Starting VOC sparse sidecar preparation"
log "project_root=${PROJECT_ROOT}"
log "dataset_root=${DATASET_ROOT}"
log "dataset_split=${DATASET_SPLIT}"
log "device=${DEVICE}"
log "batch_size=${BATCH_SIZE}"
log "num_workers=${NUM_WORKERS}"
log "output_root=${OUTPUT_ROOT}"
log "defense_config_dir=${DEFENSE_CONFIG_DIR}"
log "variants=${VARIANT_ROWS[*]}"
log "force_rebuild=${FORCE_REBUILD}"

prepared_count=0
skipped_count=0

for variant in "${VARIANT_ROWS[@]}"; do
  config_path="${DEFENSE_CONFIG_DIR}/${variant}_example.yaml"
  if [[ ! -f "${config_path}" ]]; then
    log "Missing defense template config: ${config_path}"
    exit 1
  fi

  output_dir="${OUTPUT_ROOT}/${variant}"
  mkdir -p "${output_dir}"

  for model_row in "${MODEL_ROWS[@]}"; do
    read -r family checkpoint <<< "${model_row}"
    checkpoint_stem="$(basename "${checkpoint}" .pth)"
    output_stats="${output_dir}/${checkpoint_stem}_${variant}_${DATASET_SPLIT}.pt"
    summary_path="${output_dir}/${checkpoint_stem}_${variant}_${DATASET_SPLIT}.summary.json"

    if [[ "${FORCE_REBUILD}" != "1" && -f "${output_stats}" ]]; then
      log "Skipping existing sidecar: variant=${variant} family=${family} checkpoint=${checkpoint} output=${output_stats}"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    log "Preparing sidecar: variant=${variant} family=${family} checkpoint=${checkpoint} output=${output_stats}"
    python scripts/prepare_sparse_defense.py \
      --family "${family}" \
      --checkpoint "${checkpoint}" \
      --defense-config "${config_path}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-split "${DATASET_SPLIT}" \
      --output-stats "${output_stats}" \
      --summary-path "${summary_path}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}"

    prepared_count=$((prepared_count + 1))
  done
done

log "Finished VOC sparse sidecar preparation"
log "prepared_count=${prepared_count}"
log "skipped_count=${skipped_count}"
