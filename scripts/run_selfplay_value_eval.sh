#!/bin/bash
# Self-play value training evaluation sweep.
# Tests 3 model variants at sims=1 and sims=25 vs GnuGo L1, 20 games each.
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
source "${VIRTUAL_ENV:-venv}/bin/activate"
export PYTHONPATH="$PROJECT_DIR"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GAMES=20
OPPONENT="gnugo:level=1"

AGENT_5B96F="runs/ogs_supervised_b9_5b96f/model_agent.h5"
MODEL_A1="runs/selfplay_value_b9/model_A1_vw1.0.h5"
MODEL_A2="runs/selfplay_value_b9/model_A2_lr2.5e-4.h5"
MODEL_B="runs/selfplay_value_b9/model_B_separate.h5"

echo "=== Self-Play Value Training Evaluation Sweep ==="
echo "Games: $GAMES per config, Opponent: $OPPONENT"
echo ""

echo "--- A1: Reset value head, lr=1e-3 ---"
for SIMS in 1 25; do
    echo "  sims=$SIMS:"
    python -u scripts/quick_eval_mcts.py \
        --agent "$MODEL_A1" \
        --sims "$SIMS" \
        --games "$GAMES" \
        --board-size 9 \
        --opponent "$OPPONENT" \
        --value-mode network
    echo ""
done

echo "--- A2: Reset value head, lr=2.5e-4 ---"
for SIMS in 1 25; do
    echo "  sims=$SIMS:"
    python -u scripts/quick_eval_mcts.py \
        --agent "$MODEL_A2" \
        --sims "$SIMS" \
        --games "$GAMES" \
        --board-size 9 \
        --opponent "$OPPONENT" \
        --value-mode network
    echo ""
done

echo "--- B: Separate value network (external) ---"
for SIMS in 1 25; do
    echo "  sims=$SIMS:"
    python -u scripts/quick_eval_mcts.py \
        --agent "$AGENT_5B96F" \
        --sims "$SIMS" \
        --games "$GAMES" \
        --board-size 9 \
        --opponent "$OPPONENT" \
        --value-model "$MODEL_B"
    echo ""
done

echo "=== Sweep complete ==="
