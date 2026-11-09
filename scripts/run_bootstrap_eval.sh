#!/bin/bash
# Bootstrap 3-head model evaluation sweep.
# Tests from-scratch trained model at sims=1, 25, 50 vs GnuGo L1, 20 games each.
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
source "${VIRTUAL_ENV:-venv}/bin/activate"
export PYTHONPATH="$PROJECT_DIR"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GAMES=20
OPPONENT="gnugo:level=1"

AGENT="runs/bootstrap_b9_scratch/model_bootstrap_agent.h5"

echo "=== Bootstrap 3-Head Model Evaluation Sweep ==="
echo "Agent: $AGENT"
echo "Games: $GAMES per config, Opponent: $OPPONENT"
echo ""

for SIMS in 1 25 50; do
    echo "--- sims=$SIMS ---"
    python -u scripts/quick_eval_mcts.py \
        --agent "$AGENT" \
        --sims "$SIMS" \
        --games "$GAMES" \
        --board-size 9 \
        --opponent "$OPPONENT" \
        --value-mode network
    echo ""
done

echo "--- Baseline: value_mode=zero at sims=25 ---"
python -u scripts/quick_eval_mcts.py \
    --agent "$AGENT" \
    --sims 25 \
    --games "$GAMES" \
    --board-size 9 \
    --opponent "$OPPONENT" \
    --value-mode zero
echo ""

echo "=== Sweep complete ==="
