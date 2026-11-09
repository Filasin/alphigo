#!/bin/bash
# Train a 3-head network on OGS supervised data (9x9).
#
# Prerequisites: run encode_ogs_supervised.py first to generate HDF5 chunks
# in data/ogs_supervised_b9/.
#
# Usage:
#   bash scripts/run_ogs_supervised_training.sh
#   bash scripts/run_ogs_supervised_training.sh --blocks 5 --filters 96

set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config ───────────────────────────────────────────────────────────
BOARD=9
BLOCKS=3
FILTERS=64
EPOCHS=10
BS=256
LR=1e-3
PATIENCE=3
SEED=42
EVAL_GAMES=50
DATA_DIR=data/ogs_supervised_b9

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blocks) BLOCKS="$2"; shift 2 ;;
        --filters) FILTERS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RUN_DIR="runs/ogs_supervised_b${BOARD}_${BLOCKS}b${FILTERS}f"
MODEL_DST="${RUN_DIR}/model.h5"
AGENT_DST="${RUN_DIR}/model_agent.h5"
LOG="logs/ogs_supervised_b${BOARD}_${BLOCKS}b${FILTERS}f.log"

mkdir -p "$RUN_DIR" logs

log() { echo "[ogs-sl] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

N_CHUNKS=$(ls "${DATA_DIR}"/ogs_supervised_b${BOARD}_*.h5 2>/dev/null | wc -l)
if [ "$N_CHUNKS" -eq 0 ]; then
    echo "ERROR: no HDF5 chunks found in $DATA_DIR"
    exit 1
fi

log "config: blocks=$BLOCKS filters=$FILTERS epochs=$EPOCHS lr=$LR bs=$BS"
log "data: $N_CHUNKS chunks in $DATA_DIR"

# ── Train ────────────────────────────────────────────────────────────
log "stage 1: training from scratch"
"$PY" -u scripts/train_distill_b19.py \
    --init-from none \
    --train-glob "${DATA_DIR}/ogs_supervised_b${BOARD}_*.h5" \
    --dst "$MODEL_DST" \
    --num-blocks "$BLOCKS" \
    --filters "$FILTERS" \
    --board-size "$BOARD" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --bs "$BS" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    --policy-weight 1.0 \
    --value-weight 1.0 \
    --ownership-weight 0.5 \
    --augment 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
    log "training FAILED rc=$rc"
    exit 1
fi
log "training done"

# ── Evaluate ─────────────────────────────────────────────────────────
log "stage 2: evaluating"
"$PY" -u scripts/evaluate_scaling_model.py \
    --agent "$AGENT_DST" \
    --board-size "$BOARD" \
    --eval-games "$EVAL_GAMES" \
    --seed 123 \
    --val-dir "$DATA_DIR" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
    log "evaluation FAILED rc=$rc"
    exit 1
fi
log "evaluation done"

# ── Sims sweep ───────────────────────────────────────────────────────
log "stage 3: sims sweep vs GnuGo"
for SIMS in 1 25 50 200; do
    log "sims=$SIMS"
    "$PY" -u scripts/evaluate_scaling_model.py \
        --agent "$AGENT_DST" \
        --board-size "$BOARD" \
        --eval-games "$EVAL_GAMES" \
        --eval-sims "$SIMS" \
        --seed 123 \
        --val-dir "$DATA_DIR" 2>&1 | tee -a "$LOG"
done
log "sims sweep done"

log "ALL DONE — results in $RUN_DIR and $LOG"
