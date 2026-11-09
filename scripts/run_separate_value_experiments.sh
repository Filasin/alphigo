#!/bin/bash
# Separate value network experiments — split into 4 stages.
# Each stage is ~2-3h and can be run independently. All are idempotent
# (skip-if-exists on checkpoints and CSV rows).
#
# Usage:
#   bash scripts/run_separate_value_experiments.sh <stage>
#
#   stage 1: Train binary value net              (~2-3h)
#   stage 2: Sweep binary value net              (~2h)
#   stage 3: Train score-margin value net        (~2-3h)
#   stage 4: Sweep score-margin value net        (~2h)
#   all:     Run all stages sequentially          (~10h)
#
# Example (background, survives terminal close):
#   setsid nohup bash scripts/run_separate_value_experiments.sh 1 \
#       >> logs/separate_value_experiments.log 2>&1 < /dev/null &

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

LOG=logs/separate_value_experiments.log
POLICY_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
CKPT_DIR=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints

BINARY_DST=${CKPT_DIR}/value_standalone_binary_b19.h5
BINARY_CSV=results/sims_sweep_b19_separate_binary.csv
MARGIN_DST=${CKPT_DIR}/value_standalone_margin_b19.h5
MARGIN_CSV=results/sims_sweep_b19_separate_margin.csv

log() { echo "[sep-val] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

run_sweep() {
    local ckpt=$1 csv=$2 label=$3
    log "${label}: running sims sweep"
    for sims in 1 25 50 200; do
        if [ -f "$csv" ] && grep -q "^${sims}," "$csv"; then
            log "  ${label} sims=$sims SKIP (already in CSV)"
            continue
        fi
        log "  ${label} sims=$sims starting"
        "$PY" -u scripts/bench_az_vs_gnugo_mp.py \
            --agent "$POLICY_CKPT" \
            --value-model "$ckpt" \
            --opponent gnugo:level=5 \
            --num-simulations "$sims" \
            --board-size 19 \
            --num-games 30 \
            --num-workers 4 \
            --out-csv "$csv" \
            > "logs/${label}_sweep_sims${sims}.log" 2>&1
        rc=$?
        log "  ${label} sims=$sims done rc=$rc"
    done
    log "${label}: sweep complete"
}

stage1() {
    if [ -f "$BINARY_DST" ]; then
        log "stage1: SKIP training (${BINARY_DST} exists)"
        return 0
    fi
    log "stage1: training separate value net with BINARY labels"
    "$PY" -u scripts/train_value_standalone_b19.py \
        --init-from "$POLICY_CKPT" \
        --dst "$BINARY_DST" \
        --binary \
        --seed 1 \
        --epochs 8 --bs 256 --lr 5e-4 \
        > logs/exp1_binary_train.log 2>&1
    rc=$?
    log "stage1: training done rc=$rc"
    return $rc
}

stage2() {
    if [ ! -f "$BINARY_DST" ]; then
        log "stage2: ERROR — binary checkpoint missing, run stage 1 first"
        return 1
    fi
    run_sweep "$BINARY_DST" "$BINARY_CSV" "exp1_binary"
}

stage3() {
    if [ -f "$MARGIN_DST" ]; then
        log "stage3: SKIP training (${MARGIN_DST} exists)"
        return 0
    fi
    log "stage3: training separate value net with SCORE-MARGIN labels"
    "$PY" -u scripts/train_value_standalone_b19.py \
        --init-from "$POLICY_CKPT" \
        --dst "$MARGIN_DST" \
        --seed 1 \
        --epochs 8 --bs 256 --lr 5e-4 \
        > logs/exp2_margin_train.log 2>&1
    rc=$?
    log "stage3: training done rc=$rc"
    return $rc
}

stage4() {
    if [ ! -f "$MARGIN_DST" ]; then
        log "stage4: ERROR — margin checkpoint missing, run stage 3 first"
        return 1
    fi
    run_sweep "$MARGIN_DST" "$MARGIN_CSV" "exp2_margin"
}

STAGE=${1:-all}

case "$STAGE" in
    1) stage1 ;;
    2) stage2 ;;
    3) stage3 ;;
    4) stage4 ;;
    all)
        stage1 && stage2
        stage3 && stage4
        log "ALL EXPERIMENTS DONE"
        echo ""
        echo "=== Results ==="
        echo "Exp 1 (binary):"
        cat "$BINARY_CSV" 2>/dev/null || echo "  (missing)"
        echo ""
        echo "Exp 2 (margin):"
        cat "$MARGIN_CSV" 2>/dev/null || echo "  (missing)"
        ;;
    *)
        echo "Usage: $0 {1|2|3|4|all}"
        echo "  1: Train binary value net (~2-3h)"
        echo "  2: Sweep binary value net (~2h)"
        echo "  3: Train score-margin value net (~2-3h)"
        echo "  4: Sweep score-margin value net (~2h)"
        echo "  all: Run all stages (~10h)"
        exit 1
        ;;
esac
