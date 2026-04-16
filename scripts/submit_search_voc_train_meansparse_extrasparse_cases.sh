#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_threshold_search}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
THRESHOLDS="${THRESHOLDS:-0.05 0.10 0.15 0.20 0.25 0.35 0.40}"

export RESULTS_ROOT
export DATASET_ROOT
export DATASET_SPLIT
export THRESHOLDS

FAMILIES=(
  "segmenter_vit_s"
  "segmenter_vit_s"
  "segmenter_vit_s"
  "segmenter_vit_s"
  "upernet_convnext"
  "upernet_convnext"
  "upernet_convnext"
  "upernet_convnext"
  "upernet_resnet50"
  "upernet_resnet50"
  "upernet_resnet50"
  "upernet_resnet50"
)

CHECKPOINTS=(
  "models/Segmenter_ViT_S_VOC_adv.pth"
  "models/Segmenter_ViT_S_VOC_adv.pth"
  "models/Segmenter_ViT_S_VOC_clean.pth"
  "models/Segmenter_ViT_S_VOC_clean.pth"
  "models/UperNet_ConvNext_T_VOC_adv.pth"
  "models/UperNet_ConvNext_T_VOC_adv.pth"
  "models/UperNet_ConvNext_T_VOC_clean.pth"
  "models/UperNet_ConvNext_T_VOC_clean.pth"
  "models/UperNet_ResNet50_VOC_adv.pth"
  "models/UperNet_ResNet50_VOC_adv.pth"
  "models/UperNet_ResNet50_VOC_clean.pth"
  "models/UperNet_ResNet50_VOC_clean.pth"
)

VARIANTS=(
  "meansparse"
  "extrasparse"
  "meansparse"
  "extrasparse"
  "meansparse"
  "extrasparse"
  "meansparse"
  "extrasparse"
  "meansparse"
  "extrasparse"
  "meansparse"
  "extrasparse"
)

job_ids=()

for index in "${!CHECKPOINTS[@]}"; do
  family="${FAMILIES[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  variant="${VARIANTS[$index]}"
  echo "Submitting family=${family} checkpoint=${checkpoint} variant=${variant}"
  submit_output="$(sbatch \
    --export=ALL,FAMILY="${family}",CHECKPOINT="${checkpoint}",VARIANT="${variant}" \
    "${SCRIPT_DIR}/submit_search_sparse_threshold_case.sbatch")"
  echo "${submit_output}"
  job_ids+=("$(awk '{print $4}' <<< "${submit_output}")")
done

dependency_string="$(IFS=:; echo "${job_ids[*]}")"
echo "Submitting summary job with dependency=afterok:${dependency_string}"
sbatch \
  --dependency="afterok:${dependency_string}" \
  --export=ALL,SEARCH_ROOT="${RESULTS_ROOT}",OUTPUT_DIR="${RESULTS_ROOT}" \
  "${SCRIPT_DIR}/submit_summarize_sparse_threshold_search.sbatch"
