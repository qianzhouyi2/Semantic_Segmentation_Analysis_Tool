#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

DEFENSE_CONFIG_DIR="${DEFENSE_CONFIG_DIR:-${PROJECT_ROOT}/configs/defenses}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/models/defenses}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${OUTPUT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_threshold_search}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
THRESHOLDS="${THRESHOLDS:-0.05 0.10 0.15 0.20 0.25 0.35 0.40}"
VARIANTS="${VARIANTS:-all}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ATTACK_CONFIG="${ATTACK_CONFIG:-${PROJECT_ROOT}/configs/attacks/pgd.yaml}"
BATCH_SIZE_CLEAN="${BATCH_SIZE_CLEAN:-8}"
BATCH_SIZE_ADV="${BATCH_SIZE_ADV:-2}"

export OUTPUT_ROOT
export SIDECAR_ROOT
export RESULTS_ROOT
export DEFENSE_CONFIG_DIR
export DATASET_ROOT
export DATASET_SPLIT
export THRESHOLDS
export VARIANTS
export FORCE_REBUILD
export DEVICE
export BATCH_SIZE
export NUM_WORKERS
export ATTACK_CONFIG
export BATCH_SIZE_CLEAN
export BATCH_SIZE_ADV

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Submitting sparse sidecar preparation job"
log "defense_config_dir=${DEFENSE_CONFIG_DIR}"
log "output_root=${OUTPUT_ROOT}"
log "sidecar_root=${SIDECAR_ROOT}"
log "results_root=${RESULTS_ROOT}"
log "dataset_root=${DATASET_ROOT}"
log "dataset_split=${DATASET_SPLIT}"
log "variants=${VARIANTS}"
log "thresholds=${THRESHOLDS}"
log "force_rebuild=${FORCE_REBUILD}"

mapfile -t MODEL_ROWS < <(python - <<'PY'
from src.common.voc_protocol import VOC_BASE_MODELS

for item in VOC_BASE_MODELS:
    print(f"{item['family']} {item['checkpoint']}")
PY
)

mapfile -t VARIANT_ROWS < <(python - <<'PY'
import os

from src.common.sparse_workflow import parse_sparse_variants

for item in parse_sparse_variants(os.environ.get("VARIANTS"), default=("all",)):
    print(item)
PY
)

prepare_job_ids=()
search_job_ids=()

for model_row in "${MODEL_ROWS[@]}"; do
  read -r family checkpoint <<< "${model_row}"
  for variant in "${VARIANT_ROWS[@]}"; do
    defense_template_config="${DEFENSE_CONFIG_DIR}/${variant}_example.yaml"
    log "Submitting prepare+search pipeline: family=${family} checkpoint=${checkpoint} variant=${variant}"
    prepare_job_id="$(sbatch --parsable \
      --export=ALL,FAMILY="${family}",CHECKPOINT="${checkpoint}",VARIANT="${variant}",DEFENSE_TEMPLATE_CONFIG="${defense_template_config}" \
      "${SCRIPT_DIR}/submit_prepare_sparse_sidecar_case.sbatch")"
    log "Submitted prepare job: ${prepare_job_id} for ${checkpoint} ${variant}"
    prepare_job_ids+=("${prepare_job_id}")

    search_job_id="$(sbatch --parsable \
      --dependency="afterok:${prepare_job_id}" \
      --export=ALL,FAMILY="${family}",CHECKPOINT="${checkpoint}",VARIANT="${variant}",DEFENSE_TEMPLATE_CONFIG="${defense_template_config}" \
      "${SCRIPT_DIR}/submit_search_sparse_threshold_case.sbatch")"
    log "Submitted search job: ${search_job_id} after ${prepare_job_id} for ${checkpoint} ${variant}"
    search_job_ids+=("${search_job_id}")
  done
done

dependency_string="$(IFS=:; echo "${search_job_ids[*]}")"
log "Submitting summary job with dependency=afterok:${dependency_string}"
summary_job_id="$(sbatch --parsable \
  --dependency="afterok:${dependency_string}" \
  --export=ALL,SEARCH_ROOT="${RESULTS_ROOT}",OUTPUT_DIR="${RESULTS_ROOT}" \
  "${SCRIPT_DIR}/submit_summarize_sparse_threshold_search.sbatch")"
log "Submitted summary job: ${summary_job_id}"
