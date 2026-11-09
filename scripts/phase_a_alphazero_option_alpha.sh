#!/bin/bash
# Phase A Option α: AlphaZero scaling curve under a ~24h compute budget.
#
# Two cells back-to-back, one seed, on RTX 3090 with --num-workers 4
# (book parallel_self_play pattern + tf.function batch-1 graph). At
# ~9.5s/game the wall-clock works out to roughly:
#
#   sims=25, 15 rounds × 200 games   →   8h
#   sims=50, 15 rounds × 200 games   →  16h
#                                        ---
#                                        24h
#
# This is the minimum scope for Frame 2 (the crossover thesis): a
# two-point scaling curve at fixed compute that shows AZ overtaking
# the best of {Q, PG, AC} as num_simulations grows. Per-cell sample
# size matches phase_a_train.sh's eval (200 games vs GnuGo) so the
# Elo error bars line up across methods.
#
# Defaults carried from phase_a_alphazero.sh (lr=1e-4, bs=256,
# early-stop after 15 rounds) for cross-method comparability.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-15}
GAMES=${GAMES:-200}
# vs-ref eval games: 50 is the book default. 200 cost us 86 min/round
# at sims=25 — internal eval dwarfed self-play. 50 gives a usable ±7pp
# CI on the promotion gate which is all that's load-bearing here.
EVAL_GAMES=${EVAL_GAMES:-50}
MIN_WIN_RATE=${MIN_WIN_RATE:-0.55}
LR=${LR:-1e-4}
BS=${BS:-256}
EARLY_STOP=${EARLY_STOP:-15}
WORKERS=${WORKERS:-4}
SEED=${SEED:-1}

SLUG="alphazero_alphazero_alphazero_lr${LR}_b9_seed${SEED}"

train_cell() {
    local sims="$1"
    local cell_tag="sims${sims}"
    local run_dir="runs/${SLUG}"
    local archive_dir="runs/${SLUG}_${cell_tag}"

    echo
    echo "================================================================"
    echo "  Phase A Option α: AlphaZero sims=${sims} seed=${SEED}"
    echo "  ${ROUNDS} rounds × ${GAMES} games, eval=${EVAL_GAMES}, workers=${WORKERS}"
    echo "  run dir: ${run_dir}   (archived to ${archive_dir} on completion)"
    echo "================================================================"

    # Resume support: if the per-cell archive already exists, restore it
    # under the live slug so train_rl.py picks up where this cell left off.
    if [ -d "${archive_dir}" ] && [ ! -d "${run_dir}" ]; then
        echo "  [resume] restoring ${archive_dir} -> ${run_dir}"
        mv "${archive_dir}" "${run_dir}"
    fi
    if [ -d "${run_dir}/checkpoints" ] && \
       ls "${run_dir}/checkpoints"/round_*.h5 >/dev/null 2>&1; then
        echo "  [resume] continuing from existing checkpoints in ${run_dir}/checkpoints"
    fi

    # python -u for line-buffered stdout so `tail -f` shows progress live.
    # No grep filter: grep on a pipe block-buffers, hiding progress for
    # hours. Filter at read-time with `grep ... < logs/option_alpha.log`
    # if you want the summary view.
    #
    # Also: check train_rl.py's exit status so a kill of the inner
    # process doesn't silently fall through to the next cell.
    if ! ${PY} -u train_rl.py \
        --board-size 9 \
        --encoder alphazero \
        --network alphazero \
        --method alphazero \
        --rounds "${ROUNDS}" \
        --games "${GAMES}" \
        --eval-games "${EVAL_GAMES}" \
        --min-win-rate "${MIN_WIN_RATE}" \
        --mcts-simulations "${sims}" \
        --num-workers "${WORKERS}" \
        --lr "${LR}" \
        --bs "${BS}" \
        --seed "${SEED}" \
        --early-stop-rounds "${EARLY_STOP}"; then
        echo "  [error] train_rl.py for sims=${sims} exited non-zero — aborting Option α"
        exit 1
    fi

    # Park this cell's outputs under a sims-tagged dir so cell 2 starts
    # clean rather than resuming cell 1's checkpoints (sims is not part
    # of Run.slug, so both cells would otherwise collide).
    if [ -d "${run_dir}" ]; then
        echo "  [archive] ${run_dir} -> ${archive_dir}"
        mv "${run_dir}" "${archive_dir}"
    fi
}

echo
echo "############# Phase A Option α: AZ feasibility (cell 1 only) #############"
echo "  Cell 2 (sims=50) deferred until cell 1 confirms AZ actually learns"
echo "  on this codebase. Per-round wall after the eval=50 fix is ~57 min;"
echo "  15 rounds × 57 min ≈ 14h end-to-end."

train_cell 25

echo
echo "##################### Phase A Option α (cell 1) done #####################"
echo "  Inspect results/experiments.jsonl for the per-round vs_ref curve."
echo "  If learning signal is clear, run cell 2 separately:"
echo "    bash scripts/phase_a_alphazero_option_alpha_cell2.sh   # not yet created"
