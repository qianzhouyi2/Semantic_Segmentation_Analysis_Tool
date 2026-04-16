#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_ms_sidecars
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:-${PROJECT_ROOT}/configs/defenses/meansparse_example.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/models/defenses/meansparse}"

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

mkdir -p "${OUTPUT_ROOT}"
activate_segtool
cd "${PROJECT_ROOT}"

FAMILIES=(
  "segmenter_vit_s"
  "segmenter_vit_s"
  "upernet_convnext"
  "upernet_convnext"
  "upernet_resnet50"
  "upernet_resnet50"
)

CHECKPOINTS=(
  "models/Segmenter_ViT_S_VOC_adv.pth"
  "models/Segmenter_ViT_S_VOC_clean.pth"
  "models/UperNet_ConvNext_T_VOC_adv.pth"
  "models/UperNet_ConvNext_T_VOC_clean.pth"
  "models/UperNet_ResNet50_VOC_adv.pth"
  "models/UperNet_ResNet50_VOC_clean.pth"
)

log "Starting VOC meansparse sidecar preparation"
log "project_root=${PROJECT_ROOT}"
log "dataset_root=${DATASET_ROOT}"
log "dataset_split=${DATASET_SPLIT}"
log "device=${DEVICE}"
log "output_root=${OUTPUT_ROOT}"
log "defense_config=${DEFENSE_CONFIG}"

for index in "${!CHECKPOINTS[@]}"; do
  family="${FAMILIES[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  checkpoint_stem="$(basename "${checkpoint}" .pth)"
  output_stats="${OUTPUT_ROOT}/${checkpoint_stem}_meansparse_${DATASET_SPLIT}.pt"
  summary_path="${OUTPUT_ROOT}/${checkpoint_stem}_meansparse_${DATASET_SPLIT}.summary.json"

  log "Preparing sidecar: family=${family} checkpoint=${checkpoint} output=${output_stats}"
  python scripts/prepare_sparse_defense.py \
    --family "${family}" \
    --checkpoint "${checkpoint}" \
    --defense-config "${DEFENSE_CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-split "${DATASET_SPLIT}" \
    --output-stats "${output_stats}" \
    --summary-path "${summary_path}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
done

log "Finished VOC meansparse sidecar preparation"
