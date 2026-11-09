#!/bin/bash
# Self-play value + ownership experiment.
#
# Full pipeline:
#   Stage 1: Generate self-play games (~24-30h for 10000 games at sims=25)
#   Stage 2: Train value+ownership network on self-play data (~5h)
#   Stage 3: Run sims sweep with new value net (~10h)
#
# Usage:
#   bash scripts/run_selfplay_value_experiment.sh <stage>
#   stage 1|2|3|all
#
# Example (background):
#   setsid nohup bash scripts/run_selfplay_value_experiment.sh all \
#       >> logs/selfplay_value_experiment.log 2>&1 < /dev/null &

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

LOG=logs/selfplay_value_experiment.log
POLICY_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
VALUE_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/value_standalone_margin_b19.h5
CKPT_DIR=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints

SELFPLAY_DIR=data/selfplay_b19
NEW_VALUE_DST=${CKPT_DIR}/value_selfplay_ownership_b19.h5
SWEEP_CSV=results/sims_sweep_b19_selfplay_ownership.csv

NUM_GAMES=2000
SIMS=25
WORKERS=6

log() { echo "[sp-val] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

stage1() {
    if ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 1>/dev/null 2>&1; then
        n_existing=$(ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 | wc -l)
        log "stage1: found ${n_existing} existing chunks in ${SELFPLAY_DIR}"
        log "stage1: SKIP (delete ${SELFPLAY_DIR} to regenerate)"
        return 0
    fi
    log "stage1: generating ${NUM_GAMES} self-play games at sims=${SIMS}"
    "$PY" -u scripts/generate_selfplay_data_mp.py \
        --agent "$POLICY_CKPT" \
        --value-model "$VALUE_CKPT" \
        --num-games "$NUM_GAMES" \
        --num-workers "$WORKERS" \
        --sims "$SIMS" \
        --out-dir "$SELFPLAY_DIR" \
        --chunk-size 1000 \
        --seed 42
    rc=$?
    log "stage1: done rc=$rc"
    return $rc
}

stage2() {
    if [ -f "$NEW_VALUE_DST" ]; then
        log "stage2: SKIP (${NEW_VALUE_DST} exists)"
        return 0
    fi
    if ! ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 1>/dev/null 2>&1; then
        log "stage2: ERROR — no self-play data, run stage 1 first"
        return 1
    fi
    log "stage2: training value+ownership on self-play data"
    "$PY" -u scripts/train_value_ownership_b19.py \
        --init-from "$POLICY_CKPT" \
        --train-glob "${SELFPLAY_DIR}/selfplay_b19_*.h5" \
        --dst "$NEW_VALUE_DST" \
        --seed 1 \
        --epochs 8 --bs 256 --lr 5e-4 \
        --ownership-weight 0.5
    rc=$?
    log "stage2: done rc=$rc"
    return $rc
}

stage3() {
    if [ ! -f "$NEW_VALUE_DST" ]; then
        log "stage3: ERROR — value+ownership checkpoint missing, run stage 2 first"
        return 1
    fi
    log "stage3: running sims sweep"
    for entry in 1:30 25:30 50:30 200:60 800:60; do
        sims=${entry%%:*}
        ngames=${entry##*:}
        if [ -f "$SWEEP_CSV" ] && grep -q "^${sims}," "$SWEEP_CSV"; then
            log "  sims=$sims SKIP (already in CSV)"
            continue
        fi
        log "  sims=$sims starting (n=$ngames)"
        "$PY" -u scripts/bench_az_vs_gnugo_mp.py \
            --agent "$POLICY_CKPT" \
            --value-model "$NEW_VALUE_DST" \
            --opponent gnugo:level=5 \
            --num-simulations "$sims" \
            --board-size 19 \
            --sgf-dir results/sgf_selfplay_ownership \
            --num-games "$ngames" \
            --num-workers 4 \
            --out-csv "$SWEEP_CSV"
        rc=$?
        log "  sims=$sims done rc=$rc"
    done
    log "stage3: sweep complete"
    echo ""
    echo "=== Results ==="
    cat "$SWEEP_CSV" 2>/dev/null || echo "(missing)"
}

STAGE=${1:-all}

case "$STAGE" in
    1) stage1 ;;
    2) stage2 ;;
    3) stage3 ;;
    all)
        stage1 && stage2 && stage3
        log "ALL STAGES DONE"
        ;;
    *)
        echo "Usage: $0 {1|2|3|all}"
        exit 1
        ;;
esac
