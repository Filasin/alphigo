#!/bin/bash
# AlphaZero self-play loop on 9x9 with gated promotion.
#
# Key design choices:
#   1. Force-promote rounds 0-2 (bootstrap phase — get the flywheel spinning).
#      Gate from round 3+ (55% threshold protects a real agent).
#   2. Policy-only training for round 0 (value-weight=0) so the random value
#      head can't poison the shared trunk through gradients.
#   3. sims=400, 1000 games/round for high-quality training data.
#
# Usage:
#   bash scripts/run_9x9_selfplay_loop.sh [--rounds N] [--fresh]

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
SIMS=400
GAMES_PER_ROUND=1000
EVAL_GAMES=20
EVAL_SIMS=400
WORKERS=8
BATCH_MCTS=16
TRAIN_EPOCHS=5
TRAIN_BS=256
TRAIN_LR=5e-4
ROUNDS=10
PROMOTE_THRESHOLD=55
FORCE_PROMOTE_UNTIL=3

RUN_DIR=runs/selfplay_v2_b${BOARD}
DATA_DIR=data/selfplay_v2_b${BOARD}
RESULTS_CSV=results/selfplay_v2_b${BOARD}.csv
LOG=logs/selfplay_v2_b${BOARD}.log

# ── Parse args ───────────────────────────────────────────────────────
FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --fresh) FRESH=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$FRESH" -eq 1 ]; then
    rm -rf "$RUN_DIR" "$DATA_DIR" "$RESULTS_CSV"
fi

mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[loop] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

# ── Initialize best agent (random) ──────────────────────────────────
BEST_AGENT="${RUN_DIR}/checkpoints/best_agent.h5"

if [ ! -f "$BEST_AGENT" ]; then
    log "init: creating random agent as initial best"
    "$PY" -u scripts/init_random_agent.py \
        --dst "$BEST_AGENT" \
        --board-size "$BOARD" \
        --num-blocks "$BLOCKS" \
        --filters "$FILTERS" \
        --sims "$SIMS" \
        --seed 1
    log "init: random best agent saved"
fi

# ── Write CSV header ────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,sims,n_games,a_wins,win_pct,mean_margin,promoted" > "$RESULTS_CSV"
fi

# ── Main loop ────────────────────────────────────────────────────────
for round in $(seq 0 "$((ROUNDS - 1))"); do
    ROUND_STR=$(printf '%03d' "$round")
    ROUND_DATA="${DATA_DIR}/round_${ROUND_STR}"
    CANDIDATE_MODEL="${RUN_DIR}/checkpoints/candidate_${ROUND_STR}.h5"
    CANDIDATE="${RUN_DIR}/checkpoints/candidate_${ROUND_STR}_agent.h5"

    # Skip fully completed rounds
    if grep -q "^${round}," "$RESULTS_CSV" 2>/dev/null; then
        log "round $round: SKIP (already in CSV)"
        continue
    fi

    # ── Value weight schedule ───────────────────────────────────────
    if [ "$round" -eq 0 ]; then
        VALUE_WEIGHT=0.0
    else
        VALUE_WEIGHT=1.0
    fi

    log "round $round: sims=$SIMS value_weight=$VALUE_WEIGHT best=$(basename $BEST_AGENT)"

    # ── Stage 1: Generate selfplay data with BEST agent ─────────────
    if ls "${ROUND_DATA}"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
        n_existing=$(ls "${ROUND_DATA}"/selfplay_b${BOARD}_*.h5 | wc -l)
        log "round $round stage1: SKIP ($n_existing chunks exist)"
    else
        mkdir -p "$ROUND_DATA"
        log "round $round stage1: generating $GAMES_PER_ROUND selfplay games (sims=$SIMS, agent=$(basename $BEST_AGENT))"
        "$PY" -u scripts/generate_selfplay_data_mp.py \
            --agent "$BEST_AGENT" \
            --num-games "$GAMES_PER_ROUND" \
            --num-workers "$WORKERS" \
            --sims "$SIMS" \
            --out-dir "$ROUND_DATA" \
            --chunk-size 250 \
            --seed "$((42 + round))" \
            --batch-mcts "$BATCH_MCTS" \
            --board-size "$BOARD"
        rc=$?
        if [ $rc -ne 0 ]; then
            log "round $round stage1: FAILED rc=$rc"
            exit 1
        fi
        log "round $round stage1: done"
    fi

    # ── Stage 2: Train candidate from BEST agent ────────────────────
    if [ -f "$CANDIDATE" ]; then
        log "round $round stage2: SKIP ($CANDIDATE exists)"
    else
        TRAIN_GLOBS=("${ROUND_DATA}/selfplay_b${BOARD}_*.h5")
        if [ "$round" -gt 0 ]; then
            PREV_ROUND_STR=$(printf '%03d' "$((round - 1))")
            PREV_DATA="${DATA_DIR}/round_${PREV_ROUND_STR}"
            if ls "${PREV_DATA}"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
                TRAIN_GLOBS+=("${PREV_DATA}/selfplay_b${BOARD}_*.h5")
            fi
        fi

        log "round $round stage2: training candidate (init from $(basename $BEST_AGENT), value_weight=$VALUE_WEIGHT)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from "$BEST_AGENT" \
            --train-glob "${TRAIN_GLOBS[@]}" \
            --dst "$CANDIDATE_MODEL" \
            --num-blocks "$BLOCKS" \
            --filters "$FILTERS" \
            --board-size "$BOARD" \
            --seed "$((1 + round))" \
            --epochs "$TRAIN_EPOCHS" \
            --bs "$TRAIN_BS" \
            --lr "$TRAIN_LR" \
            --policy-weight 1.0 \
            --value-weight "$VALUE_WEIGHT" \
            --ownership-weight 0.5
        rc=$?
        if [ $rc -ne 0 ]; then
            log "round $round stage2: FAILED rc=$rc"
            exit 1
        fi
        log "round $round stage2: done"
    fi

    # ── Stage 3: H2H candidate vs best → gated promotion ───────────
    log "round $round stage3: h2h candidate vs best ($EVAL_GAMES games, eval_sims=$EVAL_SIMS)"
    EVAL_TMP=$(mktemp)
    "$PY" -u scripts/head_to_head.py \
        --agent-a "$CANDIDATE" \
        --agent-b "$BEST_AGENT" \
        --num-games "$EVAL_GAMES" \
        --board-size "$BOARD" \
        --sims "$EVAL_SIMS" \
        --temperature 0.3 \
        --seed "$((100 + round))" \
        2>&1 | tee "$EVAL_TMP"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "round $round stage3: h2h FAILED rc=$rc"
        rm -f "$EVAL_TMP"
        exit 1
    fi

    A_LINE=$(grep 'A wins:' "$EVAL_TMP")
    A_WINS=$(echo "$A_LINE" | grep -o '[0-9]* (' | grep -o '[0-9]*')
    A_PCT=$(echo "$A_LINE" | grep -o '([0-9]*%)' | grep -o '[0-9]*')
    MARGIN=$(grep 'A mean margin:' "$EVAL_TMP" | grep -o '[+-][0-9.]*')
    rm -f "$EVAL_TMP"

    # Gated promotion (force-promote during bootstrap phase)
    PROMOTED="no"
    if [ "$round" -lt "$FORCE_PROMOTE_UNTIL" ]; then
        cp "$CANDIDATE" "$BEST_AGENT"
        PROMOTED="force"
        log "round $round stage3: FORCE-PROMOTED (bootstrap, ${A_PCT}%, margin=${MARGIN})"
    elif [ "$A_PCT" -ge "$PROMOTE_THRESHOLD" ]; then
        cp "$CANDIDATE" "$BEST_AGENT"
        PROMOTED="yes"
        log "round $round stage3: PROMOTED (${A_PCT}% >= ${PROMOTE_THRESHOLD}%, margin=${MARGIN})"
    else
        log "round $round stage3: NOT promoted (${A_PCT}% < ${PROMOTE_THRESHOLD}%, margin=${MARGIN})"
    fi

    echo "${round},${SIMS},${EVAL_GAMES},${A_WINS},${A_PCT},${MARGIN},${PROMOTED}" >> "$RESULTS_CSV"

    log "round $round: COMPLETE"
    echo ""
done

log "ALL $ROUNDS ROUNDS DONE"
echo ""
echo "=== Results ==="
cat "$RESULTS_CSV"
