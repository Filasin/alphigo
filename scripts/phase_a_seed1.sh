#!/bin/bash
# Phase A — single seed (seed=1) wrapper. Runs PG/Q/AC, then AlphaZero,
# then a post-hoc GnuGo L1 eval sweep over the trained checkpoints.
# ~15-20h training + ~50min post-hoc on RTX 3090. Set-and-forget overnight
# scout before committing to the full 3-seed sweep.
#
# Logs:
#   logs/phase_a_seed1_pgqac.log
#   logs/phase_a_seed1_az.log
#   logs/phase_a_seed1_posthoc.log
#
# Re-run is safe: each method's train_rl.py auto-resumes from existing
# checkpoints in runs/<slug>_seed1/, and SEEDS=1 only touches seed-1 dirs.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

export SEEDS=1
export EARLY_STOP=${EARLY_STOP:-15}

echo "$(date -Iseconds) — Phase A seed 1 launch (early-stop=${EARLY_STOP})"

echo
echo "[1/3] PG/Q/AC seed=1 → logs/phase_a_seed1_pgqac.log"
bash scripts/phase_a_train.sh > logs/phase_a_seed1_pgqac.log 2>&1
rc1=$?
echo "  PG/Q/AC done (rc=$rc1) at $(date +%H:%M:%S)"

echo
echo "[2/3] AlphaZero seed=1 → logs/phase_a_seed1_az.log"
bash scripts/phase_a_alphazero.sh > logs/phase_a_seed1_az.log 2>&1
rc2=$?
echo "  AlphaZero done (rc=$rc2) at $(date +%H:%M:%S)"

echo
echo "[3/3] Post-hoc GnuGo L1 eval → logs/phase_a_seed1_posthoc.log"
${ALPHIGO_PYTHON:-python} scripts/posthoc_eval_seed1.py \
    > logs/phase_a_seed1_posthoc.log 2>&1
rc3=$?
echo "  Post-hoc done (rc=$rc3) at $(date +%H:%M:%S)"

echo
echo "$(date -Iseconds) — Phase A seed 1 complete (PG/Q/AC rc=$rc1, AZ rc=$rc2, posthoc rc=$rc3)"
