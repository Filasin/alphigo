#!/bin/bash
# Data scaling experiment: train models on increasing subsets of bootstrap data,
# evaluate each to produce a sample-efficiency curve.
#
# Usage:
#   bash scripts/run_scaling_experiment.sh
#
# Prerequisites:
#   - Bootstrap data generated in data/bootstrap_b9/
#   - Converted to HDF5 in data/selfplay_bootstrap_b9/
#   - GnuGo available in gtp_agents/

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
BS=1024
LR=5e-4
SEED=42

DATA_DIR=data/selfplay_bootstrap_b9
RUN_DIR=runs/scaling_experiment_b9
RESULTS_CSV=results/scaling_experiment_b9.csv
LOG=logs/scaling_experiment_b9.log

# Number of games per subset (each chunk ≈ 5000 positions ≈ ~77 games)
# We control by number of HDF5 chunks used for training
# Each chunk = ~5000 positions from convert_npy_to_h5.py
# 1K games ≈ 65K pos ≈ 13 chunks
# 5K games ≈ 325K pos ≈ 65 chunks
# 10K games ≈ 650K pos ≈ 130 chunks
# 25K games ≈ 1.6M pos ≈ 325 chunks
# 50K games ≈ 3.2M pos ≈ 650 chunks
# 80K games ≈ 5.2M pos ≈ 1040 chunks
GAME_COUNTS=(1000 5000 10000 25000 50000 80000)

EVAL_GAMES=50
EVAL_SIMS=200

mkdir -p "$RUN_DIR" logs results

log() { echo "[scaling] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

# ── Write CSV header ────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "games,positions,chunks_used,policy_loss,value_loss,ownership_loss,total_loss,vs_random_winpct,vs_gnugo1_winpct,vs_gnugo5_winpct,policy_entropy_first_move" > "$RESULTS_CSV"
fi

# ── Count available chunks ──────────────────────────────────────────
ALL_CHUNKS=($(ls "$DATA_DIR"/selfplay_b${BOARD}_*.h5 2>/dev/null | sort))
TOTAL_CHUNKS=${#ALL_CHUNKS[@]}
if [ "$TOTAL_CHUNKS" -eq 0 ]; then
    log "ERROR: no HDF5 chunks found in $DATA_DIR"
    log "Run: python scripts/convert_npy_to_h5.py --input-dir data/bootstrap_b9 --tag mcts --board-size 9 --out-dir $DATA_DIR --chunk-size 5000"
    exit 1
fi
log "Found $TOTAL_CHUNKS chunks in $DATA_DIR"

# ── Estimate positions per chunk ────────────────────────────────────
# Read first chunk to get positions count
FIRST_CHUNK_POS=$("$PY" -c "
import h5py
with h5py.File('${ALL_CHUNKS[0]}', 'r') as h5:
    print(len(h5['zero_experience/encoded_states']))
")
POS_PER_CHUNK=${FIRST_CHUNK_POS:-5000}
log "Positions per chunk: ~$POS_PER_CHUNK"

# ── Main loop ────────────────────────────────────────────────────────
for N_GAMES in "${GAME_COUNTS[@]}"; do
    SUBSET_TAG="g${N_GAMES}"
    MODEL_PATH="$RUN_DIR/model_${SUBSET_TAG}.h5"
    AGENT_PATH="$RUN_DIR/model_${SUBSET_TAG}_agent.h5"

    # Skip if already evaluated
    if grep -q "^${N_GAMES}," "$RESULTS_CSV" 2>/dev/null; then
        log "$SUBSET_TAG: SKIP (already in CSV)"
        continue
    fi

    # Calculate how many chunks we need
    TARGET_POS=$((N_GAMES * 65))
    N_CHUNKS=$(( (TARGET_POS + POS_PER_CHUNK - 1) / POS_PER_CHUNK ))
    if [ "$N_CHUNKS" -gt "$TOTAL_CHUNKS" ]; then
        SHORTFALL=$(( (N_CHUNKS - TOTAL_CHUNKS) * 100 / N_CHUNKS ))
        if [ "$SHORTFALL" -le 5 ]; then
            log "$SUBSET_TAG: capping from $N_CHUNKS to $TOTAL_CHUNKS chunks (${SHORTFALL}% short)"
            N_CHUNKS=$TOTAL_CHUNKS
        else
            log "$SUBSET_TAG: SKIP (need $N_CHUNKS chunks, only $TOTAL_CHUNKS available)"
            continue
        fi
    fi

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "$SUBSET_TAG: training on $N_CHUNKS chunks (~${TARGET_POS} positions)"

    # ── Stage 1: Train ──────────────────────────────────────────────
    if [ ! -f "$AGENT_PATH" ]; then
        # Build glob pattern for first N chunks
        CHUNK_LIST=""
        for i in $(seq 0 $((N_CHUNKS - 1))); do
            if [ -n "$CHUNK_LIST" ]; then CHUNK_LIST="$CHUNK_LIST "; fi
            CHUNK_LIST="${CHUNK_LIST}${ALL_CHUNKS[$i]}"
        done

        log "$SUBSET_TAG stage1: training (epochs=$EPOCHS, lr=$LR)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from none \
            --train-glob $CHUNK_LIST \
            --dst "$MODEL_PATH" \
            --num-blocks "$BLOCKS" \
            --filters "$FILTERS" \
            --board-size "$BOARD" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --bs "$BS" \
            --lr "$LR" \
            --policy-weight 1.0 \
            --value-weight 1.0 \
            --ownership-weight 0.5
        rc=$?
        if [ $rc -ne 0 ]; then
            log "$SUBSET_TAG stage1: FAILED rc=$rc"
            exit 1
        fi
        log "$SUBSET_TAG stage1: done"
    else
        log "$SUBSET_TAG stage1: SKIP (model exists)"
    fi

    # ── Stage 2: Evaluate ───────────────────────────────────────────
    log "$SUBSET_TAG stage2: evaluating"
    EVAL_OUT=$("$PY" -u scripts/evaluate_scaling_model.py \
        --agent "$AGENT_PATH" \
        --board-size "$BOARD" \
        --eval-games "$EVAL_GAMES" \
        --eval-sims "$EVAL_SIMS" \
        --seed 123)
    rc=$?
    if [ $rc -ne 0 ]; then
        log "$SUBSET_TAG stage2: FAILED rc=$rc"
        log "$SUBSET_TAG stage2: output=$EVAL_OUT"
        exit 1
    fi

    # Parse evaluation output (format: positions,policy_loss,value_loss,ownership_loss,total_loss,vs_random,vs_gnugo,entropy)
    ACTUAL_POS=$(echo "$EVAL_OUT" | grep "^RESULT:" | cut -d: -f2)
    log "$SUBSET_TAG stage2: $ACTUAL_POS"
    echo "${N_GAMES},${ACTUAL_POS}" >> "$RESULTS_CSV"

    log "$SUBSET_TAG: COMPLETE"
    echo ""
done

log "ALL SUBSETS DONE"
echo ""
echo "=== Results ==="
cat "$RESULTS_CSV"
