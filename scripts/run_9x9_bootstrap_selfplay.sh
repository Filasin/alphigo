#!/bin/bash
# Bootstrapped self-play experiment: network-guided MCTS self-play
# initialized from the 80K rollout bootstrap model.
#
# Tests whether replacing random rollouts with neural network evaluation
# in MCTS produces qualitatively better training data, breaking the
# policy loss ceiling of 3.75 from the data scaling experiment.
#
# Differences from run_9x9_selfplay_loop.sh:
#   - --bootstrap: initialize from trained model (not random)
#   - sims=800 (matches C++ MCTS depth)
#   - value_weight=1.0 from round 0 (bootstrap has trained value head)
#   - --full-init: preserve all weights when training candidates
#   - Gated promotion: 52% threshold, 50 h2h games
#   - Stage 4: absolute evaluation (vs GnuGo, policy loss, entropy)
#
# Usage:
#   bash scripts/run_9x9_bootstrap_selfplay.sh \
#       --bootstrap runs/scaling_experiment_b9/model_g80000_agent.h5
#
# Benchmark mode (override parallelism):
#   bash scripts/run_9x9_bootstrap_selfplay.sh \
#       --bootstrap runs/scaling_experiment_b9/model_g80000_agent.h5 \
#       --games 20 --rounds 1 --batch-mcts 32 --workers 10

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
SIMS=800
GAMES_PER_ROUND=1000
H2H_GAMES=10
EVAL_GAMES=50
WORKERS=8
BATCH_MCTS=32
TRAIN_EPOCHS=2
TRAIN_BS=256
TRAIN_LR=5e-4
ROUNDS=15
PROMOTE_THRESHOLD=50
BOOTSTRAP_DATA=data/selfplay_bootstrap_b${BOARD}
BOOTSTRAP_CHUNKS=50
H2H_SIMS=200

RUN_DIR=runs/selfplay_bootstrap_loop_b9
DATA_DIR=data/selfplay_bootstrap_loop_b9
RESULTS_CSV=results/selfplay_bootstrap_loop_b9.csv
LOG=logs/selfplay_bootstrap_loop_b9.log

# ── Parse args ───────────────────────────────────────────────────────
BOOTSTRAP=""
FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --sims) SIMS="$2"; shift 2 ;;
        --batch-mcts) BATCH_MCTS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --games) GAMES_PER_ROUND="$2"; shift 2 ;;
        --fresh) FRESH=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$FRESH" -eq 1 ]; then
    rm -rf "$RUN_DIR" "$DATA_DIR" "$RESULTS_CSV"
fi

mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[bootstrap] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

log "config: sims=$SIMS games=$GAMES_PER_ROUND rounds=$ROUNDS workers=$WORKERS batch_mcts=$BATCH_MCTS"
log "config: promote_threshold=$PROMOTE_THRESHOLD h2h_games=$H2H_GAMES eval_games=$EVAL_GAMES"

# ── Initialize best agent ─────────────────────────────────────────────
BEST_AGENT="${RUN_DIR}/checkpoints/best_agent.h5"

if [ ! -f "$BEST_AGENT" ]; then
    if [ -z "$BOOTSTRAP" ]; then
        echo "ERROR: --bootstrap PATH required (no existing best_agent found)"
        exit 1
    fi
    if [ ! -f "$BOOTSTRAP" ]; then
        echo "ERROR: bootstrap agent not found: $BOOTSTRAP"
        exit 1
    fi
    log "init: copying bootstrap agent $BOOTSTRAP"
    cp "$BOOTSTRAP" "$BEST_AGENT"
    log "init: bootstrap agent ready"
fi

# ── Write CSV header ────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,sims,n_games,policy_loss,value_loss,ownership_loss,total_loss,vs_random,vs_gnugo1,vs_gnugo5,entropy,h2h_winpct,h2h_margin,promoted" > "$RESULTS_CSV"
fi

# ── Main loop ────────────────────────────────────────────────────────
for round in $(seq 0 "$((ROUNDS - 1))"); do
    ROUND_STR=$(printf '%03d' "$round")
    ROUND_DATA="${DATA_DIR}/round_${ROUND_STR}"
    CANDIDATE_MODEL="${RUN_DIR}/checkpoints/candidate_${ROUND_STR}.h5"
    CANDIDATE="${RUN_DIR}/checkpoints/candidate_${ROUND_STR}_agent.h5"

    if grep -q "^${round}," "$RESULTS_CSV" 2>/dev/null; then
        log "round $round: SKIP (already in CSV)"
        continue
    fi

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "round $round: sims=$SIMS best=$(basename $BEST_AGENT)"

    # ── Stage 1: Generate selfplay data ────────────────────────────
    if ls "${ROUND_DATA}"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
        n_existing=$(ls "${ROUND_DATA}"/selfplay_b${BOARD}_*.h5 | wc -l)
        log "round $round stage1: SKIP ($n_existing chunks exist)"
    else
        mkdir -p "$ROUND_DATA"
        log "round $round stage1: generating $GAMES_PER_ROUND games (sims=$SIMS, workers=$WORKERS, batch_mcts=$BATCH_MCTS)"
        STAGE1_START=$(date +%s)
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
        STAGE1_END=$(date +%s)
        STAGE1_ELAPSED=$((STAGE1_END - STAGE1_START))
        if [ $rc -ne 0 ]; then
            log "round $round stage1: FAILED rc=$rc after ${STAGE1_ELAPSED}s"
            exit 1
        fi
        log "round $round stage1: done (${STAGE1_ELAPSED}s, ~$(( GAMES_PER_ROUND * 3600 / (STAGE1_ELAPSED + 1) )) games/hour)"
    fi

    # ── Stage 2: Train candidate (full init from best) ─────────────
    if [ -f "$CANDIDATE" ]; then
        log "round $round stage2: SKIP ($CANDIDATE exists)"
    else
        BOOTSTRAP_SAMPLE=$(ls "${BOOTSTRAP_DATA}"/selfplay_b${BOARD}_*.h5 | shuf -n "$BOOTSTRAP_CHUNKS" --random-source=<(echo "$((42 + round))"))
        TRAIN_GLOBS=()
        for f in $BOOTSTRAP_SAMPLE; do
            TRAIN_GLOBS+=("$f")
        done
        N_SELFPLAY=0
        for r in $(seq 0 "$round"); do
            R_STR=$(printf '%03d' "$r")
            R_DATA="${DATA_DIR}/round_${R_STR}"
            if ls "${R_DATA}"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
                TRAIN_GLOBS+=("${R_DATA}/selfplay_b${BOARD}_*.h5")
                N_SELFPLAY=$((N_SELFPLAY + $(ls "${R_DATA}"/selfplay_b${BOARD}_*.h5 | wc -l)))
            fi
        done

        log "round $round stage2: training (full-init, ${BOOTSTRAP_CHUNKS} bootstrap + ${N_SELFPLAY} selfplay chunks, ${TRAIN_EPOCHS} epochs)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from "$BEST_AGENT" \
            --full-init \
            --augment \
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
            --value-weight 1.0 \
            --ownership-weight 0.5
        rc=$?
        if [ $rc -ne 0 ]; then
            log "round $round stage2: FAILED rc=$rc"
            exit 1
        fi
        log "round $round stage2: done"
    fi

    # ── Stage 3: H2H candidate vs best → gated promotion (50%) ─────
    log "round $round stage3: h2h candidate vs best ($H2H_GAMES games, sims=$H2H_SIMS)"
    EVAL_TMP=$(mktemp)
    "$PY" -u scripts/head_to_head.py \
        --agent-a "$CANDIDATE" \
        --agent-b "$BEST_AGENT" \
        --num-games "$H2H_GAMES" \
        --board-size "$BOARD" \
        --sims "$H2H_SIMS" \
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
    A_PCT=$(echo "$A_LINE" | grep -o '([0-9]*%)' | grep -o '[0-9]*')
    MARGIN=$(grep 'A mean margin:' "$EVAL_TMP" | grep -o '[+-][0-9.]*')
    rm -f "$EVAL_TMP"

    PROMOTED="yes"
    cp "$CANDIDATE" "$BEST_AGENT"
    log "round $round stage3: PROMOTED (h2h=${A_PCT}%, always-promote with replay buffer)"

    # ── Stage 4: Absolute evaluation of best_agent ─────────────────
    log "round $round stage4: absolute evaluation of best_agent"
    ABS_OUT=$("$PY" -u scripts/evaluate_scaling_model.py \
        --agent "$BEST_AGENT" \
        --board-size "$BOARD" \
        --eval-games "$EVAL_GAMES" \
        --seed 123 \
        --val-dir "$DATA_DIR/round_${ROUND_STR}")
    rc=$?
    if [ $rc -ne 0 ]; then
        log "round $round stage4: eval FAILED rc=$rc (continuing with h2h data)"
        EVAL_METRICS="-1.0000,-1.0000,-1.0000,-1.0000,-1.0,-1.0,-1.0,-1.0000"
    else
        EVAL_METRICS=$(echo "$ABS_OUT" | grep "^RESULT:" | cut -d: -f2)
    fi

    echo "${round},${SIMS},${GAMES_PER_ROUND},${EVAL_METRICS},${A_PCT},${MARGIN},${PROMOTED}" >> "$RESULTS_CSV"

    log "round $round: COMPLETE (promoted=$PROMOTED, h2h=${A_PCT}%, margin=${MARGIN})"
    echo ""
done

log "ALL $ROUNDS ROUNDS DONE"
echo ""
echo "=== Results ==="
column -t -s, "$RESULTS_CSV" 2>/dev/null || cat "$RESULTS_CSV"
