#!/bin/bash
# Position diagnostic: compare value heads on a live game.
# Run after selfplay + control experiments are done (or standalone).
source "${VIRTUAL_ENV:-venv}/bin/activate"
SITE=${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"
cd "$(dirname "$0")/.."

RUN=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints
AGENT=$RUN/r0_joint_smargin_b19.h5
VALUE_PRO=$RUN/value_standalone_margin_b19.h5
VALUE_SP=$RUN/value_selfplay_ownership_b19.h5

LOG=logs/diagnose_positions.log

echo "[diagnose] $(date) Starting position diagnostic" | tee $LOG

# Game 1: AZ as Black, sims=25 (fast game for trajectory)
echo "[diagnose] $(date) Game 1: AZ=Black, sims=25" | tee -a $LOG
python scripts/diagnose_positions.py \
    --agent $AGENT \
    --value-pro $VALUE_PRO \
    --value-sp $VALUE_SP \
    --opponent gnugo:level=5 \
    --board-size 19 \
    --sims 25 \
    --az-color black \
    --probe-sims 200 \
    --num-probes 5 \
    --top-positions 8 \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG

# Game 2: AZ as White, sims=25
echo "[diagnose] $(date) Game 2: AZ=White, sims=25" | tee -a $LOG
python scripts/diagnose_positions.py \
    --agent $AGENT \
    --value-pro $VALUE_PRO \
    --value-sp $VALUE_SP \
    --opponent gnugo:level=5 \
    --board-size 19 \
    --sims 25 \
    --az-color white \
    --probe-sims 200 \
    --num-probes 5 \
    --top-positions 8 \
    2>&1 | tee -a $LOG

echo "[diagnose] $(date) DONE" | tee -a $LOG
