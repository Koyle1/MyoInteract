#!/usr/bin/env bash
#SBATCH --job-name=steering-5ids-20trials
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Run from the MyoInteract directory in your activated Python environment:
#   sbatch jobs/evaluate_steering_5ids.sbatch --checkpoint /path/to/checkpoints
# Add your cluster's account/partition on the sbatch command if required.
# Optional: set CONDA_ENV=myointeract_env to use conda run.
# Preview the 100 assignments locally without a checkpoint or simulator:
#   bash jobs/evaluate_steering_5ids.sbatch --dry-run --output /tmp/steering-plan
#
# Requires a non-vision steering task with working evaluation hooks. The
# pointing task in this checkout cannot collect steering-law trajectories.
# See evaluate_steering.py --help for the task metadata requirements.
# This script schedules attempts; failed trials count toward the 20 per ID.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
else
    PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
cd -- "${PROJECT_DIR}"

runner="${PROJECT_DIR}/jobs/evaluate_steering.py"
if [[ ! -f "${runner}" ]]; then
    echo "Cannot find ${runner}. Submit from MyoInteract or set PROJECT_DIR." >&2
    exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run_id="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
args=(
    --ids 5 10 15 20 30
    --trials-per-id 20
    --path-length "${PATH_LENGTH_M:-0.20}"
    --seed "${EVAL_SEED:-123}"
    --output "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/steering_eval_${run_id}}"
)
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    args+=(--checkpoint "${CHECKPOINT_PATH}")
fi

command=("${PYTHON_BIN:-python3}" "${runner}" "${args[@]}" "$@")
if [[ -n "${CONDA_ENV:-}" ]]; then
    command=(conda run --no-capture-output -n "${CONDA_ENV}" "${command[@]}")
fi
exec "${command[@]}"
