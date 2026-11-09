#!/bin/bash
# Train 10b128f 3-head model from scratch on C++ MCTS bootstrap data (80K games).
# Produces the round-0 seed agent for the large-model Gumbel self-play loop.
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DST_DIR=runs/bootstrap_b9_10b128f
mkdir -p "$DST_DIR"

"$PY" -u scripts/train_distill_b19.py \
    --init-from none \
    --train-glob 'data/bootstrap_b9_h5/selfplay_b9_*.h5' \
    --dst "$DST_DIR/model_bootstrap.h5" \
    --num-blocks 10 \
    --filters 128 \
    --board-size 9 \
    --augment \
    --epochs 10 \
    --bs 1024 \
    --lr 5e-4 \
    --patience 3 \
    --mega-chunks 4 \
    --policy-weight 1.0 --value-weight 1.0 --ownership-weight 0.5 \
    --seed 1
