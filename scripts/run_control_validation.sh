#!/bin/bash
# Control validation for self-play experiment results.
# Runs: (1) sims=200 sweep with ORIGINAL value net (baseline check)
#        (2) Value output comparison between original and self-play models
#
# Usage:
#   bash scripts/run_control_validation.sh
#
# Requires GPU environment (LD_LIBRARY_PATH etc.) to be set up.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

POLICY=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
ORIG_VALUE=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/value_standalone_margin_b19.h5
SP_VALUE=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/value_selfplay_ownership_b19.h5
CONTROL_CSV=results/sims_sweep_b19_control_original.csv
LOG=logs/control_validation.log

log() { echo "[control] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

# ── Check 1: Control sweep with original value net ──────────────
log "CHECK 1: sims=200 control sweep with original value net (20 games)"
"$PY" -u scripts/bench_az_vs_gnugo_mp.py \
    --agent "$POLICY" \
    --value-model "$ORIG_VALUE" \
    --opponent gnugo:level=5 \
    --num-simulations 200 \
    --board-size 19 \
    --num-games 20 \
    --num-workers 4 \
    --out-csv "$CONTROL_CSV"
log "CHECK 1 done (rc=$?)"
log "Control CSV:"
cat "$CONTROL_CSV" 2>/dev/null | tee -a "$LOG"

# ── Check 2: Value prediction comparison ────────────────────────
log "CHECK 2: value output comparison on 200 self-play positions"
"$PY" -u scripts/compare_value_outputs.py \
    --orig-model "$ORIG_VALUE" \
    --sp-model "$SP_VALUE" \
    --data "data/selfplay_b19/selfplay_b19_0000.h5" \
    --num-positions 200
log "CHECK 2 done (rc=$?)"

log "ALL CONTROL CHECKS COMPLETE"
