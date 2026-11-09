#!/bin/bash
# Cross-method bench vs GnuGo level 1 (and Random as anchor).
#
# Benches the best agent of each method at the same 5-round budget.
# All vs gnugo:level=1 at N=100 games (color-swapped via run.py benchmark).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}
OPP=${OPP:-gnugo:level=1}

# Lines: <method>:<agent_file>:<label>
ENTRIES=(
    "q:rl_agents/9x9_oneplane_book_2layer_q_seed1/round_005.h5:Q  oneplane+book_2layer (seed 1)"
    "q:rl_agents/9x9_oneplane_book_2layer_q_seed2/round_005.h5:Q  oneplane+book_2layer (seed 2)"
    "q:rl_agents/9x9_oneplane_book_2layer_q_seed3/round_005.h5:Q  oneplane+book_2layer (seed 3)"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_seed1/round_005.h5:AC oneplane+book_2layer (seed 1)"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_seed2/round_005.h5:AC oneplane+book_2layer (seed 2)"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_seed3/round_005.h5:AC oneplane+book_2layer (seed 3)"
    "ac:rl_agents/9x9_oneplane_small_nobn_ac_seed1/round_005.h5:AC oneplane+small_nobn (seed 1)"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_vw0.25/round_005.h5:AC oneplane+book_2layer (vw=0.25, best sens)"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_eps0.1/round_005.h5:AC oneplane+book_2layer (eps=0.1)"
    "q-dqn:rl_agents/9x9_oneplane_small_q_dqn/round_005.h5:Q-DQN oneplane+small (seed 1)"
    "pg:rl_agents/9x9_oneplane_small_pg_lr1e4/round_005.h5:PG oneplane+small lr=1e-4 (seed 1)"
    "dl:checkpoints/mcts_oneplane_small_adam_b9_model_epoch_5.h5:Supervised mcts_oneplane_small_b9 epoch 5"
    "dl:checkpoints/mcts_oneplane_small_adam_b9_model_epoch_6.h5:Supervised mcts_oneplane_small_b9 epoch 6"
)

echo "===== Method vs ${OPP} (N=${N_GAMES}) ====="
printf "%-50s  %s\n" "agent" "win%"
printf "%.s-" {1..70}; printf "\n"

for entry in "${ENTRIES[@]}"; do
    method="${entry%%:*}"
    rest="${entry#*:}"
    agent_file="${rest%%:*}"
    label="${rest#*:}"

    printf "%-50s  " "${label}"
    if [ ! -f "${agent_file}" ]; then
        printf "(missing %s)\n" "${agent_file}"
        continue
    fi
    wins=$(${PY} run.py benchmark \
        --black "${method}:agent_file=${agent_file}" --white "${OPP}" \
        --board-size 9 -n "${N_GAMES}" 2>/dev/null \
        | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
    if [ -z "${wins}" ]; then
        printf "err\n"
    else
        pct=$(( 100 * wins / N_GAMES ))
        printf "%3d%%  (%s wins)\n" "${pct}" "${wins}"
    fi
done
