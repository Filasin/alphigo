#!/bin/bash
# Ownership-as-value proxy evaluation sweep.
# Compares three value modes at sims=25 vs GnuGo L1, 20 games each.
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
source "${VIRTUAL_ENV:-venv}/bin/activate"
export PYTHONPATH="$PROJECT_DIR"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

AGENT="${1:-runs/ogs_supervised_b9_5b96f/model_agent.h5}"
SIMS=25
GAMES=20
OPPONENT="gnugo:level=1"

echo "=== Ownership Proxy Evaluation Sweep ==="
echo "Agent: $AGENT"
echo "Sims: $SIMS, Games: $GAMES, Opponent: $OPPONENT"
echo ""

for MODE in network ownership zero; do
    echo "--- value_mode=$MODE ---"
    python -u scripts/quick_eval_mcts.py \
        --agent "$AGENT" \
        --sims "$SIMS" \
        --games "$GAMES" \
        --board-size 9 \
        --opponent "$OPPONENT" \
        --value-mode "$MODE"
    echo ""
done

echo "=== Sweep complete ==="
