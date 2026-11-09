#!/bin/bash
# Post-fix rerun: re-evaluate all 9x9 MCTS cells after the PUCT
# sign-inversion fix (2026-06-11), then restart bootstrap training.
#
# Original (pre-fix) numbers for comparison, all vs GnuGo L1, N=20:
#   5b96f network  sims=1: 24%   sims=25: 0%
#   5b96f ownership sims=25: 5%
#   5b96f zero      sims=25: 15%   (sign-insensitive sanity anchor)
#   A1 network      sims=1: 20%  sims=25: 0%
#   A2 network      sims=25: 0%
#   B external      sims=25: 0%
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
source "${VIRTUAL_ENV:-venv}/bin/activate"
export PYTHONPATH="$PROJECT_DIR"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GAMES=20
OPP="gnugo:level=1"
AGENT="runs/ogs_supervised_b9_5b96f/model_agent.h5"
MODEL_A1="runs/selfplay_value_b9/model_A1_vw1.0.h5"
MODEL_A2="runs/selfplay_value_b9/model_A2_lr2.5e-4.h5"
MODEL_B="runs/selfplay_value_b9/model_B_separate.h5"

run_cell () {
    local label="$1"; shift
    echo ""
    echo "--- $label ---"
    python -u scripts/quick_eval_mcts.py "$@" \
        --games "$GAMES" --board-size 9 --opponent "$OPP"
}

echo "=== Post-fix rerun (PUCT sign fix) $(date) ==="

# 5b96f: value-source comparison + sims sweep
run_cell "5b96f network sims=1 (was 24%)"  --agent "$AGENT" --sims 1   --value-mode network
run_cell "5b96f zero sims=25 (anchor, was 15%)" --agent "$AGENT" --sims 25 --value-mode zero
run_cell "5b96f network sims=25 (was 0%)"  --agent "$AGENT" --sims 25  --value-mode network
run_cell "5b96f ownership sims=25 (was 5%)" --agent "$AGENT" --sims 25 --value-mode ownership
run_cell "5b96f network sims=50 (new)"     --agent "$AGENT" --sims 50  --value-mode network
run_cell "5b96f network sims=100 (new)"    --agent "$AGENT" --sims 100 --value-mode network

# Self-play value-trained models
run_cell "A1 network sims=1 (was 20%)"     --agent "$MODEL_A1" --sims 1  --value-mode network
run_cell "A1 network sims=25 (was 0%)"     --agent "$MODEL_A1" --sims 25 --value-mode network
run_cell "A2 network sims=25 (was 0%)"     --agent "$MODEL_A2" --sims 25 --value-mode network
run_cell "B external sims=25 (was 0%)"     --agent "$AGENT" --sims 25 --value-model "$MODEL_B"

echo ""
echo "=== Rerun complete $(date) ==="
echo ""
echo "=== Restarting bootstrap training ==="
bash scripts/run_bootstrap_train.sh
