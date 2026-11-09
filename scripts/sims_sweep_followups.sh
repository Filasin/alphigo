#!/bin/bash
# Tier-1 follow-up sweeps for the "MCTS amplifies value-head noise"
# claim. Runs after the initial sims_sweep.sh on r0_value_trained.h5
# has produced results/sims_sweep_r0_value_trained.csv.
#
# Experiment A: mechanism check — random-value-head sweep.
# Re-runs sims_sweep on bootstrap r0 (SL trunk+policy + RANDOM value
# head, never fine-tuned). Expected pattern: degradation grows with
# sims, more steeply than the trained-value sweep. Output side-by-side
# gives the two-curve panel "as value quality drops, search hurts more".
#
# Experiment B: tighter CIs at headline points.
# Re-runs r0_value_trained.h5 at sims ∈ {1, 25, 200} with N=200 games
# each. Cuts Wilson CI from ~±14pp to ~±7pp at these three anchor
# points.
#
# Sequential by default; comment out one block to skip.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

# ── Experiment A: random-value-head sweep ──────────────────────────
echo
echo "############# Follow-up A: random-value-head sweep #############"
AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints/round_000.h5 \
    OUT_CSV=results/sims_sweep_r0_random_value.csv \
    bash scripts/sims_sweep.sh

# ── Experiment B: tighter CIs at sims=1, 25, 200 (N=200) ───────────
echo
echo "############# Follow-up B: tighter CIs on r0_value_trained #############"
AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints/r0_value_trained.h5 \
    OUT_CSV=results/sims_sweep_r0_value_trained_N200.csv \
    N_GAMES=200 \
    SIMS_LIST="1 25 200" \
    bash scripts/sims_sweep.sh

echo
echo "############# follow-ups done #############"
echo "  Inspect:"
echo "    results/sims_sweep_r0_value_trained.csv  (initial, N=50)"
echo "    results/sims_sweep_r0_random_value.csv   (experiment A)"
echo "    results/sims_sweep_r0_value_trained_N200.csv  (experiment B)"
