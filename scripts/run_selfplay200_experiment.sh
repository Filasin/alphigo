#!/bin/bash
# Self-play value experiment at sims=200.
#
# Tests whether generating self-play data at the EVALUATION depth
# (sims=200) fixes the distribution mismatch that caused the sims=25
# self-play value net to fail.
#
# Pipeline:
#   Stage 1:  Generate self-play games at sims=200 (~20-26h)
#   Stage 2a: Train value+ownership net on that data (~20min)
#   Stage 2b: Train policy+value+ownership (distillation) on that data (~20min)
#   Stage 3a: Sims sweep benchmark with value-only model (~6-10h)
#   Stage 3b: Sims sweep benchmark with distilled model (~6-10h)
#
# Usage:
#   bash scripts/run_selfplay200_experiment.sh [1|2a|2b|3a|3b|all]

set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

# CUDA libs are pip-installed under nvidia packages in the venv
NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LOG=logs/selfplay200_experiment.log
mkdir -p logs results

POLICY_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
VALUE_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/value_standalone_margin_b19.h5
CKPT_DIR=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints

SELFPLAY_DIR=data/selfplay_b19_sims200
NEW_VALUE_DST=${CKPT_DIR}/value_selfplay200_ownership_b19.h5
DISTILL_DST=${CKPT_DIR}/distill_selfplay200_pvo_b19.h5
DISTILL_AGENT=${CKPT_DIR}/distill_selfplay200_pvo_b19_agent.h5
SWEEP_CSV=results/sims_sweep_b19_selfplay200_ownership.csv
SWEEP_DISTILL_CSV=results/sims_sweep_b19_selfplay200_distill.csv

NUM_GAMES=2000
SIMS=200
WORKERS=4

log() { echo "[sp200] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

stage1() {
    if ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 1>/dev/null 2>&1; then
        n_existing=$(ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 | wc -l)
        log "stage1: found ${n_existing} existing chunks in ${SELFPLAY_DIR}"
        log "stage1: SKIP (delete ${SELFPLAY_DIR} to regenerate)"
        return 0
    fi
    mkdir -p "$SELFPLAY_DIR"
    log "stage1: generating ${NUM_GAMES} self-play games at sims=${SIMS} (workers=${WORKERS})"
    "$PY" -u scripts/generate_selfplay_data_mp.py \
        --agent "$POLICY_CKPT" \
        --value-model "$VALUE_CKPT" \
        --num-games "$NUM_GAMES" \
        --num-workers "$WORKERS" \
        --sims "$SIMS" \
        --out-dir "$SELFPLAY_DIR" \
        --chunk-size 200 \
        --seed 42 \
        --batch-mcts 8
    rc=$?
    log "stage1: done rc=$rc"
    return $rc
}

stage2a() {
    if [ -f "$NEW_VALUE_DST" ]; then
        log "stage2a: SKIP (${NEW_VALUE_DST} exists)"
        return 0
    fi
    if ! ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 1>/dev/null 2>&1; then
        log "stage2a: ERROR — no self-play data, run stage 1 first"
        return 1
    fi
    log "stage2a: training value+ownership on sims=200 self-play data"
    "$PY" -u scripts/train_value_ownership_b19.py \
        --init-from "$POLICY_CKPT" \
        --train-glob "${SELFPLAY_DIR}/selfplay_b19_*.h5" \
        --dst "$NEW_VALUE_DST" \
        --seed 1 \
        --epochs 8 --bs 256 --lr 5e-4 \
        --ownership-weight 0.5
    rc=$?
    log "stage2a: done rc=$rc"
    return $rc
}

stage2b() {
    if [ -f "$DISTILL_DST" ]; then
        log "stage2b: SKIP (${DISTILL_DST} exists)"
        return 0
    fi
    if ! ls "${SELFPLAY_DIR}"/selfplay_b19_*.h5 1>/dev/null 2>&1; then
        log "stage2b: ERROR — no self-play data, run stage 1 first"
        return 1
    fi
    log "stage2b: training policy+value+ownership (distillation) on sims=200 data"
    "$PY" -u scripts/train_distill_b19.py \
        --init-from "$POLICY_CKPT" \
        --train-glob "${SELFPLAY_DIR}/selfplay_b19_*.h5" \
        --dst "$DISTILL_DST" \
        --seed 1 \
        --epochs 8 --bs 256 --lr 5e-4 \
        --policy-weight 1.0 --ownership-weight 0.5
    rc=$?
    log "stage2b: done rc=$rc"
    return $rc
}

stage3a() {
    if [ ! -f "$NEW_VALUE_DST" ]; then
        log "stage3a: ERROR — value+ownership checkpoint missing, run stage 2a first"
        return 1
    fi
    log "stage3a: sims sweep with value-only model"
    for entry in 1:30 25:30 50:30 200:30 800:30; do
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
            --num-games "$ngames" \
            --num-workers 4 \
            --out-csv "$SWEEP_CSV"
        rc=$?
        log "  sims=$sims done rc=$rc"
    done
    log "stage3a: sweep complete"
    echo ""
    echo "=== Results (value-only) ==="
    cat "$SWEEP_CSV" 2>/dev/null || echo "(missing)"
}

stage3b() {
    if [ ! -f "$DISTILL_AGENT" ]; then
        log "stage3b: ERROR — distilled agent missing, run stage 2b first"
        return 1
    fi
    log "stage3b: sims sweep with distilled policy+value model"
    for entry in 1:30 25:30 50:30 200:30 800:30; do
        sims=${entry%%:*}
        ngames=${entry##*:}
        if [ -f "$SWEEP_DISTILL_CSV" ] && grep -q "^${sims}," "$SWEEP_DISTILL_CSV"; then
            log "  sims=$sims SKIP (already in CSV)"
            continue
        fi
        log "  sims=$sims starting (n=$ngames)"
        "$PY" -u scripts/bench_az_vs_gnugo_mp.py \
            --agent "$DISTILL_AGENT" \
            --opponent gnugo:level=5 \
            --num-simulations "$sims" \
            --board-size 19 \
            --num-games "$ngames" \
            --num-workers 4 \
            --out-csv "$SWEEP_DISTILL_CSV"
        rc=$?
        log "  sims=$sims done rc=$rc"
    done
    log "stage3b: sweep complete"
    echo ""
    echo "=== Results (distilled) ==="
    cat "$SWEEP_DISTILL_CSV" 2>/dev/null || echo "(missing)"
}

STAGE=${1:-all}

case "$STAGE" in
    1) stage1 ;;
    2a) stage2a ;;
    2b) stage2b ;;
    3a) stage3a ;;
    3b) stage3b ;;
    all)
        stage1 && stage2a && stage2b && stage3a && stage3b
        log "ALL STAGES DONE"
        ;;
    *)
        echo "Usage: $0 {1|2a|2b|3a|3b|all}"
        exit 1
        ;;
esac
