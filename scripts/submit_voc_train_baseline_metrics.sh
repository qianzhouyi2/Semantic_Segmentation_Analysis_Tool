#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_baseline_metrics}"
SEARCH_ROOT="${SEARCH_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_threshold_search_rerun}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
ATTACK_CONFIG="${ATTACK_CONFIG:-${PROJECT_ROOT}/configs/attacks/pgd.yaml}"
JOB_NICE="${JOB_NICE:-0}"

mapfile -t MODEL_ROWS < <(python - <<'PY'
from src.common.voc_protocol import VOC_BASE_MODELS

for item in VOC_BASE_MODELS:
    print(f"{item['name']} {item['family']} {item['checkpoint']}")
PY
)

job_ids=()

for model_row in "${MODEL_ROWS[@]}"; do
  read -r name family checkpoint <<< "${model_row}"
  model_id="baseline__${name}"
  clean_output="${RESULTS_ROOT}/clean/${model_id}"
  attack_output="${RESULTS_ROOT}/attacks/pgd/${model_id}"

  clean_job_id="$(sbatch --parsable \
    --nice="${JOB_NICE}" \
    --export=ALL,MODE=clean,MODEL_ID="${model_id}",FAMILY="${family}",CHECKPOINT="${checkpoint}",OUTPUT_DIR="${clean_output}",DATASET_ROOT="${DATASET_ROOT}",DATASET_SPLIT="${DATASET_SPLIT}",NUM_WORKERS=4,BATCH_SIZE=8,DEVICE=cuda \
    scripts/submit_voc_eval_case.sbatch)"
  echo "Submitted clean baseline job ${clean_job_id} for ${model_id}"
  job_ids+=("${clean_job_id}")

  attack_job_id="$(sbatch --parsable \
    --nice="${JOB_NICE}" \
    --export=ALL,MODE=attack,MODEL_ID="${model_id}",FAMILY="${family}",CHECKPOINT="${checkpoint}",OUTPUT_DIR="${attack_output}",DATASET_ROOT="${DATASET_ROOT}",DATASET_SPLIT="${DATASET_SPLIT}",ATTACK_CONFIG="${ATTACK_CONFIG}",NUM_WORKERS=4,BATCH_SIZE=2,DEVICE=cuda \
    scripts/submit_voc_eval_case.sbatch)"
  echo "Submitted PGD baseline job ${attack_job_id} for ${model_id}"
  job_ids+=("${attack_job_id}")
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
summary_job_id="$(sbatch --parsable \
  --nice="${JOB_NICE}" \
  --dependency="afterok:${dependency}" \
  --export=ALL,SEARCH_ROOT="${SEARCH_ROOT}",BASELINE_ROOT="${RESULTS_ROOT}",OUTPUT_DIR="${SEARCH_ROOT}/summary" \
  scripts/submit_voc_train_baseline_metrics_summary.sbatch)"
echo "Submitted baseline summary job ${summary_job_id}"
