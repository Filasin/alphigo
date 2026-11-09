#!/bin/bash
# Seed-variance characterisation: 3 more explicit-seed retrains + sweeps.
# Each cycle: train ~5h, then 4-cell sweep ~2h. Total ~21h sequential.
#
# Waits for the current seed-2 sims=200 cell to finish before starting.
# Re-runs the missing seed-2 sims=25 cell first (the one that crashed).

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

LOG=logs/seed_variance_chain.log
STATUS=logs/STATUS.md
SEED1_AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
SRC_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/round_000.h5
SEED2_CSV=results/sims_sweep_b19_smargin_seed2.csv

log() { echo "[chain] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }
write_status() {
    cat > "$STATUS" <<EOF
# Seed-variance chain status

Updated: $(date -Iseconds)

$1
EOF
}

write_status "**STAGE 0**: waiting for current seed-2 sweep to finish"
log "stage 0: waiting for seed-2 sweep wrapper PID 25034 to exit"

# Wait for the parent wrapper script to finish.
while kill -0 25034 2>/dev/null; do
    sleep 60
done
log "stage 0 done: seed-2 sweep finished"

# Re-run the missing seed-2 sims=25 cell with the robust mp script — only
# if it isn't already in the CSV.
SEED2_AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19_seed2.h5
if [ -f "$SEED2_CSV" ] && grep -q "^25," "$SEED2_CSV"; then
    log "stage 0b skipped: seed-2 sims=25 already in $SEED2_CSV"
else
    write_status "**STAGE 0b**: rerunning crashed seed-2 sims=25 cell"
    log "stage 0b: rerunning seed-2 sims=25 (crashed earlier)"
    "$PY" -u scripts/bench_az_vs_gnugo_mp.py \
        --agent "$SEED2_AGENT" \
        --opponent gnugo:level=5 \
        --num-simulations 25 \
        --board-size 19 \
        --num-games 30 \
        --num-workers 4 \
        --out-csv "$SEED2_CSV" \
        > logs/seed2_sims25_rerun.log 2>&1
    rc=$?
    log "stage 0b done rc=$rc"
fi

# Three more seeded retrain + sweep cycles.
for SEED in 3 4 5; do
    AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19_seed${SEED}.h5
    CSV=results/sims_sweep_b19_smargin_seed${SEED}.csv

    write_status "**STAGE ${SEED}a**: training seed=${SEED} (target: $AGENT)"
    log "stage ${SEED}a: training seed=${SEED}"

    if [ ! -f "$AGENT" ]; then
        "$PY" -u scripts/train_value_joint_b19.py \
            --src "$SRC_CKPT" \
            --train-glob "data/sgf_value_buffer_b19_v2_*.h5" \
            --val-chunk "data/sgf_value_buffer_b19_v2_0015.h5" \
            --dst "$AGENT" \
            --epochs 8 --bs 256 --lr 5e-4 --alpha 0.5 \
            --seed "$SEED" \
            > "logs/seed${SEED}_train.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] || [ ! -f "$AGENT" ]; then
            log "SEED-${SEED} TRAIN FAILED rc=$rc. Skipping this seed's sweep."
            write_status "**WARNING** seed=${SEED} train failed (rc=$rc). Moving to next seed."
            continue
        fi
        log "stage ${SEED}a done: $AGENT saved"
    else
        log "stage ${SEED}a skipped: $AGENT already exists"
    fi

    write_status "**STAGE ${SEED}b**: sweeping seed=${SEED}"
    log "stage ${SEED}b: sweep seed=${SEED}"
    # Per-cell skip: only run sims cells not yet in the CSV.
    for sims in 1 25 50 200; do
        if [ -f "$CSV" ] && grep -q "^${sims}," "$CSV"; then
            log "  seed=${SEED} sims=$sims SKIP (already in $CSV)"
            continue
        fi
        log "  seed=${SEED} sims=$sims starting"
        "$PY" -u scripts/bench_az_vs_gnugo_mp.py \
            --agent "$AGENT" \
            --opponent gnugo:level=5 \
            --num-simulations "$sims" \
            --board-size 19 \
            --num-games 30 \
            --num-workers 4 \
            --out-csv "$CSV" \
            > "logs/seed${SEED}_sweep_sims${sims}.log" 2>&1
        rc=$?
        log "  seed=${SEED} sims=$sims done rc=$rc"
    done
    log "stage ${SEED}b done"
done

# Final status.
{
    echo "# Seed-variance chain — COMPLETE"
    echo
    echo "Updated: $(date -Iseconds)"
    echo
    for s in 1 2 3 4 5; do
        if [ "$s" = "1" ]; then
            F=results/sims_sweep_b19_smargin.csv
        else
            F=results/sims_sweep_b19_smargin_seed${s}.csv
        fi
        echo "## seed=${s}"
        echo '```'
        cat "$F" 2>/dev/null || echo "(missing)"
        echo '```'
        echo
    done
} > "$STATUS"
log "DONE"
