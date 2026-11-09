#!/bin/bash
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p runs/ogs_supervised_b9_3b64f logs

${ALPHIGO_PYTHON:-python} -u scripts/train_distill_b19.py \
    --init-from none \
    --train-glob "data/ogs_supervised_b9/ogs_supervised_b9_*.h5" \
    --dst runs/ogs_supervised_b9_3b64f/model.h5 \
    --num-blocks 3 --filters 64 --board-size 9 \
    --seed 42 --epochs 10 --bs 256 --lr 1e-3 \
    --patience 3 --augment \
    --mega-chunks 5 \
    --policy-weight 1.0 --value-weight 1.0 --ownership-weight 0.5
