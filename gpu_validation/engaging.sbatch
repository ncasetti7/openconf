#!/usr/bin/env bash
#SBATCH --job-name=openconf-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=openconf-gpu-%j.out
#SBATCH --error=openconf-gpu-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted with sbatch." >&2
    exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not allocate a GPU." >&2
    exit 2
fi

uv sync --extra gpu --torch-backend=cu128
.venv/bin/python gpu_validation/validate.py --output "openconf-gpu-${SLURM_JOB_ID}.json"
