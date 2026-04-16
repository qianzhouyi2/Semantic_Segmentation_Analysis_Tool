#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_thr_search
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
ATTACK_CONFIG="${ATTACK_CONFIG:-${PROJECT_ROOT}/configs/attacks/pgd.yaml}"
THRESHOLDS="${THRESHOLDS:-0.05,0.10,0.15,0.20,0.25,0.35,0.40}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/reports/voc_train_threshold_search}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${PROJECT_ROOT}/models/defenses}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE_CLEAN="${BATCH_SIZE_CLEAN:-8}"
BATCH_SIZE_ADV="${BATCH_SIZE_ADV:-2}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-4}"

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
  local variant="$1"
  local family="$2"
  local checkpoint="$3"
  local checkpoint_stem
  checkpoint_stem="$(basename "${checkpoint}" .pth)"
  local output_dir="${SIDECAR_ROOT}/${variant}"
  local stats_path="${output_dir}/${checkpoint_stem}_${variant}_${DATASET_SPLIT}.pt"
  local summary_path="${output_dir}/${checkpoint_stem}_${variant}_${DATASET_SPLIT}.summary.json"
  local config_path="${PROJECT_ROOT}/configs/defenses/${variant}_example.yaml"
  mkdir -p "${output_dir}"
  if [[ ! -f "${stats_path}" ]]; then
    log "Preparing sidecar variant=${variant} family=${family} checkpoint=${checkpoint}"
    CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/prepare_sparse_defense.py" \
      --family "${family}" \
      --checkpoint "${PROJECT_ROOT}/${checkpoint}" \
      --defense-config "${config_path}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-split "${DATASET_SPLIT}" \
      --output-stats "${stats_path}" \
      --summary-path "${summary_path}" \
      --batch-size "${CALIB_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device cuda
  fi
  printf '%s\n' "${stats_path}"
}

run_case() {
  local gpu_id="$1"
  local family="$2"
  local checkpoint="$3"
  local variant="$4"
  local stats_path="$5"
  local checkpoint_stem
  checkpoint_stem="$(basename "${checkpoint}" .pth)"
  local output_dir="${RESULTS_ROOT}/${checkpoint_stem}/${variant}"
  log "GPU ${gpu_id}: search family=${family} checkpoint=${checkpoint_stem} variant=${variant}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python "${PROJECT_ROOT}/scripts/search_sparse_thresholds.py" \
    --family "${family}" \
    --checkpoint "${PROJECT_ROOT}/${checkpoint}" \
    --variant "${variant}" \
    --stats-path "${stats_path}" \
    --attack-config "${ATTACK_CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-split "${DATASET_SPLIT}" \
    --thresholds "${THRESHOLDS}" \
    --output-dir "${output_dir}" \
    --batch-size-clean "${BATCH_SIZE_CLEAN}" \
    --batch-size-adv "${BATCH_SIZE_ADV}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda
}

worker_loop() {
  local gpu_id="$1"
  shift
  local case_indices=("$@")
  local index family checkpoint variant stats_path
  for index in "${case_indices[@]}"; do
    family="${FAMILIES[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"
    variant="${VARIANTS[$index]}"
    stats_path="${STAT_PATHS[$index]}"
    run_case "${gpu_id}" "${family}" "${checkpoint}" "${variant}" "${stats_path}"
  done
}

activate_segtool
cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}"

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

STAT_PATHS=()
for index in "${!CHECKPOINTS[@]}"; do
  STAT_PATHS+=("$(ensure_sidecar "${VARIANTS[$index]}" "${FAMILIES[$index]}" "${CHECKPOINTS[$index]}")")
done

GPU0_CASES=(0 3 4 7 8 11)
GPU1_CASES=(1 2 5 6 9 10)

log "Starting two-GPU threshold search"
log "dataset_root=${DATASET_ROOT} split=${DATASET_SPLIT} thresholds=${THRESHOLDS}"

worker_loop 0 "${GPU0_CASES[@]}" > "${RESULTS_ROOT}/gpu0.log" 2>&1 &
pid0=$!
worker_loop 1 "${GPU1_CASES[@]}" > "${RESULTS_ROOT}/gpu1.log" 2>&1 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

python "${PROJECT_ROOT}/scripts/summarize_sparse_threshold_search.py" \
  --search-root "${RESULTS_ROOT}" \
  --output-dir "${RESULTS_ROOT}"

log "Finished two-GPU threshold search"
