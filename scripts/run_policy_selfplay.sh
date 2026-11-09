#!/bin/bash
source "${VIRTUAL_ENV:-venv}/bin/activate"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

NVIDIA_LIB=${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib"

POLICY_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
VALUE_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/value_standalone_margin_b19.h5
OUT_DIR=data/selfplay_b19_sims200_policy

mkdir -p "$OUT_DIR"

echo "=== Generating 200 games at sims=200 WITH policy targets ==="
python -u scripts/generate_selfplay_data_mp.py \
    --agent "$POLICY_CKPT" \
    --value-model "$VALUE_CKPT" \
    --num-games 200 \
    --num-workers 4 \
    --sims 200 \
    --out-dir "$OUT_DIR" \
    --chunk-size 100 \
    --seed 99 \
    --batch-mcts 8

echo "=== Selfplay done ==="
