#!/bin/bash
source "${VIRTUAL_ENV:-venv}/bin/activate"
SITE=${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"
export PYTHONPATH="$(pwd)"
cd "$(dirname "$0")/.."

RUN=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints
LOG=logs/diagnose_positions.log

echo "" > $LOG

echo "=== $(date) Game 1: AZ=Black, sims=25 ===" | tee -a $LOG
python scripts/diagnose_positions.py \
    --agent $RUN/r0_joint_smargin_b19.h5 \
    --value-pro $RUN/value_standalone_margin_b19.h5 \
    --value-sp $RUN/value_selfplay_ownership_b19.h5 \
    --opponent gnugo:level=5 \
    --board-size 19 \
    --sims 25 \
    --az-color black \
    --probe-sims 200 \
    --num-probes 5 \
    --top-positions 8 \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG

echo "=== $(date) Game 2: AZ=White, sims=25 ===" | tee -a $LOG
python scripts/diagnose_positions.py \
    --agent $RUN/r0_joint_smargin_b19.h5 \
    --value-pro $RUN/value_standalone_margin_b19.h5 \
    --value-sp $RUN/value_selfplay_ownership_b19.h5 \
    --opponent gnugo:level=5 \
    --board-size 19 \
    --sims 25 \
    --az-color white \
    --probe-sims 200 \
    --num-probes 5 \
    --top-positions 8 \
    2>&1 | tee -a $LOG

echo "=== $(date) DONE ===" | tee -a $LOG
