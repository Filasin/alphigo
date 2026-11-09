#!/bin/bash
# Phase C (AlphaZero): num_simulations sensitivity sweep.
#
# Fixes LR at the best-from-Phase-B value (default 1e-4) and sweeps
# --mcts-simulations across {10, 25, 50, 100}.  Each sims value lands
# in the SAME Run dir (Run.slug doesn't encode num_simulations), so
# we tag results in JSONL via the round key only.  To keep separate
# checkpoints per simulations setting, the script uses an isolated
# subdir per simulations value (ceiling-isolation, not via Run).
#
# Set BEST_LR=<value> from Phase B before running.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-10}
GAMES=${GAMES:-200}
EVAL_GAMES=${EVAL_GAMES:-30}
BEST_LR=${BEST_LR:-1e-4}
SIMS_LIST=${SIMS_LIST:-"10 25 50 100"}

train_one_sims() {
    local sims="$1"
    local slug="alphazero_alphazero_alphazero_lr${BEST_LR}_b9"
    local home="runs/${slug}"
    local stash="runs/_phase_c_stash_sims${sims}"

    # Stash any prior dir from a different sims setting so this run
    # gets a fresh canonical home; restore the stash for "${sims}" if it
    # exists. This keeps each sims value's checkpoints isolated.
    if [ -d "${home}" ] && [ ! -d "${stash}" ]; then
        # Only stash if no current marker matches
        if [ ! -f "${home}/_sims" ] || [ "$(cat ${home}/_sims)" != "${sims}" ]; then
            local prev_sims="unknown"
            [ -f "${home}/_sims" ] && prev_sims=$(cat ${home}/_sims)
            mv "${home}" "runs/_phase_c_stash_sims${prev_sims}"
        fi
    fi
    if [ -d "${stash}" ]; then
        mv "${stash}" "${home}"
        echo "  [resume] restored Phase C run for sims=${sims}"
    fi

    ${PY} train_rl.py \
        --board-size 9 \
        --encoder alphazero \
        --network alphazero \
        --method alphazero \
        --rounds "${ROUNDS}" \
        --games "${GAMES}" \
        --eval-games "${EVAL_GAMES}" \
        --mcts-simulations "${sims}" \
        --lr "${BEST_LR}" \
        --bs 256 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'

    # Mark the dir with this sims value so the next iteration knows.
    mkdir -p "${home}"
    echo "${sims}" > "${home}/_sims"
}

for sims in ${SIMS_LIST}; do
    echo "================================================================"
    echo "=== Phase C AZ sims=${sims} lr=${BEST_LR}"
    echo "================================================================"
    train_one_sims "${sims}"
done

echo
echo "##################### Phase C AlphaZero done #####################"
