#!/bin/bash
# Train 2-head model from scratch on C++ MCTS bootstrap data (80K games, 5.17M positions).
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
source "${VIRTUAL_ENV:-venv}/bin/activate"
export PYTHONPATH="$PROJECT_DIR"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u scripts/train_distill_b19.py \
    --init-from none \
    --train-glob 'data/bootstrap_b9_h5/selfplay_b9_*.h5' \
    --dst runs/bootstrap_b9_scratch/model_bootstrap.h5 \
    --num-blocks 8 \
    --filters 64 \
    --board-size 9 \
    --heads 2 \
    --augment \
    --epochs 8 \
    --bs 1024 \
    --lr 5e-4 \
    --patience 3 \
    --mega-chunks 4 \
    --seed 1
