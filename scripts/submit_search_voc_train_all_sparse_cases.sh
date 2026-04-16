#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_threshold_search}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
THRESHOLDS="${THRESHOLDS:-0.05 0.10 0.15 0.20 0.25 0.35 0.40}"
JOB_DEPENDENCY="${JOB_DEPENDENCY:-}"
VARIANTS="${VARIANTS:-$(python - <<'PY'
from src.models.sparse import SPARSE_DEFENSE_CHOICES

print(" ".join(SPARSE_DEFENSE_CHOICES))
PY
)}"

export RESULTS_ROOT
export DATASET_ROOT
export DATASET_SPLIT
export THRESHOLDS
export VARIANTS
export JOB_DEPENDENCY

mapfile -t MODEL_ROWS < <(python - <<'PY'
from src.common.voc_protocol import VOC_BASE_MODELS

for item in VOC_BASE_MODELS:
    print(f"{item['family']} {item['checkpoint']}")
PY
)

mapfile -t VARIANT_ROWS < <(python - <<'PY'
import os

from src.common.sparse_workflow import parse_sparse_variants

for item in parse_sparse_variants(os.environ.get("VARIANTS")):
    print(item)
PY
)

job_ids=()
submit_args=()
if [[ -n "${JOB_DEPENDENCY}" ]]; then
  echo "Using upstream dependency=${JOB_DEPENDENCY}"
  submit_args+=(--dependency="${JOB_DEPENDENCY}")
fi

for model_row in "${MODEL_ROWS[@]}"; do
  read -r family checkpoint <<< "${model_row}"
  for variant in "${VARIANT_ROWS[@]}"; do
    echo "Submitting family=${family} checkpoint=${checkpoint} variant=${variant}"
    submit_output="$(sbatch \
      "${submit_args[@]}" \
      --export=ALL,FAMILY="${family}",CHECKPOINT="${checkpoint}",VARIANT="${variant}" \
      "${SCRIPT_DIR}/submit_search_sparse_threshold_case.sbatch")"
    echo "${submit_output}"
    job_ids+=("$(awk '{print $4}' <<< "${submit_output}")")
  done
done

dependency_string="$(IFS=:; echo "${job_ids[*]}")"
echo "Submitting summary job with dependency=afterok:${dependency_string}"
sbatch \
  --dependency="afterok:${dependency_string}" \
  --export=ALL,SEARCH_ROOT="${RESULTS_ROOT}",OUTPUT_DIR="${RESULTS_ROOT}" \
  "${SCRIPT_DIR}/submit_summarize_sparse_threshold_search.sbatch"
