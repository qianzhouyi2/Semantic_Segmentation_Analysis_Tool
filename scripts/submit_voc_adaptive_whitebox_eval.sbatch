#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --job-name=voc_adaptive_wb
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

FAMILY="${FAMILY:?FAMILY is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

ATTACK_CONFIG="${ATTACK_CONFIG:-configs/attacks/pgd_adaptive_sparse_strong.yaml}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:-}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
EPSILON_RADIUS_255="${EPSILON_RADIUS_255:-}"
EPSILON_RADIUS_255_SWEEP="${EPSILON_RADIUS_255_SWEEP:-2 4 8}"
ATTACK_BACKWARD_MODE="${ATTACK_BACKWARD_MODE:-bpda_ste}"
NUM_RESTARTS="${NUM_RESTARTS:-5}"
EOT_ITERS="${EOT_ITERS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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

mkdir -p "${OUTPUT_DIR}"
activate_segtool
cd "${PROJECT_ROOT}"

cmd=(
  python scripts/run_attack.py
  --attack-config "${ATTACK_CONFIG}"
  --family "${FAMILY}"
  --checkpoint "${CHECKPOINT}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --attack-backward-mode "${ATTACK_BACKWARD_MODE}"
  --num-restarts "${NUM_RESTARTS}"
  --eot-iters "${EOT_ITERS}"
)

if [[ -n "${DEFENSE_CONFIG}" ]]; then
  cmd+=(--defense-config "${DEFENSE_CONFIG}")
fi

if [[ -n "${EPSILON_RADIUS_255}" ]]; then
  cmd+=(--epsilon-radius-255 "${EPSILON_RADIUS_255}")
else
  # shellcheck disable=SC2206
  sweep_values=(${EPSILON_RADIUS_255_SWEEP})
  cmd+=(--epsilon-radius-255-sweep "${sweep_values[@]}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_ARGS})
  cmd+=("${extra_args[@]}")
fi

log "Starting adaptive white-box evaluation"
log "project_root=${PROJECT_ROOT}"
log "family=${FAMILY}"
log "checkpoint=${CHECKPOINT}"
log "attack_config=${ATTACK_CONFIG}"
log "defense_config=${DEFENSE_CONFIG:-<none>}"
log "output_dir=${OUTPUT_DIR}"
log "attack_backward_mode=${ATTACK_BACKWARD_MODE} num_restarts=${NUM_RESTARTS} eot_iters=${EOT_ITERS}"
if [[ -n "${EPSILON_RADIUS_255}" ]]; then
  log "epsilon_radius_255=${EPSILON_RADIUS_255}"
else
  log "epsilon_radius_255_sweep=${EPSILON_RADIUS_255_SWEEP}"
fi

"${cmd[@]}"

log "Finished adaptive white-box evaluation"
