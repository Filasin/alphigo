#!/bin/bash
# Phase 1: GPU benchmark for bootstrapped self-play experiment.
#
# Sweeps (batch_mcts, workers) at sims=800 to find optimal parallelism
# on RTX 3060 + Ryzen 5 2600. Generates 20 games per configuration
# and reports games/hour.
#
# Usage:
#   bash scripts/run_benchmark_selfplay.sh \
#       --bootstrap runs/scaling_experiment_b9/model_g80000_agent.h5

set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config ───────────────────────────────────────────────────────────
BOARD=9
SIMS=800
BENCH_GAMES=20

BATCH_MCTS_VALUES=(8 16 32)
WORKERS_VALUES=(4 8 10)

BENCH_DIR=runs/benchmark_selfplay_b9
LOG=logs/benchmark_selfplay_b9.log

# ── Parse args ───────────────────────────────────────────────────────
BOOTSTRAP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
        --games) BENCH_GAMES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$BOOTSTRAP" ]; then
    echo "ERROR: --bootstrap PATH required"
    exit 1
fi
if [ ! -f "$BOOTSTRAP" ]; then
    echo "ERROR: bootstrap agent not found: $BOOTSTRAP"
    exit 1
fi

mkdir -p "$BENCH_DIR" logs

log() { echo "[bench] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

log "Benchmark: sims=$SIMS, games=$BENCH_GAMES"
log "Bootstrap: $BOOTSTRAP"
log ""

# ── Results header ───────────────────────────────────────────────────
RESULTS_FILE="${BENCH_DIR}/results.csv"
echo "batch_mcts,workers,games,wall_seconds,games_per_hour" > "$RESULTS_FILE"

printf "%-12s %-10s %-8s %-12s %-12s\n" "batch_mcts" "workers" "games" "seconds" "games/hour"
printf "%-12s %-10s %-8s %-12s %-12s\n" "----------" "-------" "-----" "-------" "----------"

# ── Sweep ────────────────────────────────────────────────────────────
for BM in "${BATCH_MCTS_VALUES[@]}"; do
    for W in "${WORKERS_VALUES[@]}"; do
        TAG="bm${BM}_w${W}"
        OUT_DIR="${BENCH_DIR}/${TAG}"

        if [ -d "$OUT_DIR" ] && ls "${OUT_DIR}"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
            log "$TAG: SKIP (data exists)"
            continue
        fi

        mkdir -p "$OUT_DIR"
        log "$TAG: batch_mcts=$BM workers=$W sims=$SIMS games=$BENCH_GAMES"

        START=$(date +%s)
        "$PY" -u scripts/generate_selfplay_data_mp.py \
            --agent "$BOOTSTRAP" \
            --num-games "$BENCH_GAMES" \
            --num-workers "$W" \
            --sims "$SIMS" \
            --out-dir "$OUT_DIR" \
            --chunk-size "$BENCH_GAMES" \
            --seed 42 \
            --batch-mcts "$BM" \
            --board-size "$BOARD" \
            2>&1 | tail -5
        rc=$?
        END=$(date +%s)
        ELAPSED=$((END - START))

        if [ $rc -ne 0 ]; then
            log "$TAG: FAILED rc=$rc"
            GPH="FAILED"
        else
            if [ "$ELAPSED" -gt 0 ]; then
                GPH=$(( BENCH_GAMES * 3600 / ELAPSED ))
            else
                GPH=999999
            fi
            echo "${BM},${W},${BENCH_GAMES},${ELAPSED},${GPH}" >> "$RESULTS_FILE"
        fi

        printf "%-12s %-10s %-8s %-12s %-12s\n" "$BM" "$W" "$BENCH_GAMES" "${ELAPSED}s" "$GPH"
        log "$TAG: ${ELAPSED}s → ${GPH} games/hour"
        echo ""
    done
done

log ""
log "=== Benchmark Results ==="
column -t -s, "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"

log ""
log "Choose the row with highest games_per_hour."
log "Then calculate: rounds = (budget_hours * games_per_hour) / 1000"
log "Example: 60h budget, 100 games/hour → (60*100)/1000 = 6 rounds"
log "  (plus ~1h/round for training+eval overhead)"
