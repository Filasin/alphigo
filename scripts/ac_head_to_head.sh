#!/bin/bash
# Direct head-to-head between AC and other methods (N=100).
# AC plays Black; opponent plays White; --swap-colors via run.py benchmark.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

# Each row: <ac-file>:<opponent-spec>:<label>
ROWS=(
    "rl_agents/9x9_oneplane_book_2layer_ac_seed1/round_005.h5:q:agent_file=rl_agents/9x9_oneplane_book_2layer_q_seed1/round_005.h5:AC vs Q (oneplane+book_2layer, seed1)"
    "rl_agents/9x9_oneplane_book_2layer_ac_seed1/round_005.h5:pg:agent_file=rl_agents/9x9_oneplane_small_pg_lr1e4/round_005.h5:AC vs PG-lr1e-4 (oneplane+small)"
    "rl_agents/9x9_oneplane_book_2layer_ac_seed1/round_005.h5:q-dqn:agent_file=rl_agents/9x9_oneplane_small_q_dqn/round_005.h5:AC vs Q-DQN (oneplane+small)"
    "rl_agents/9x9_oneplane_book_2layer_ac_seed1/round_005.h5:dl:agent_file=checkpoints/mcts_oneplane_small_adam_b9_model_epoch_5.h5:AC vs Supervised (mcts ep 5)"
    "rl_agents/9x9_oneplane_book_2layer_ac_vw0.25/round_005.h5:q:agent_file=rl_agents/9x9_oneplane_book_2layer_q_seed1/round_005.h5:AC-best (vw=0.25) vs Q seed1"
)

echo "===== AC head-to-head (N=${N_GAMES}, AC plays Black, swap-colors on) ====="
printf "%-60s  %s\n" "matchup" "AC win%"
printf "%.s-" {1..80}; printf "\n"

for row in "${ROWS[@]}"; do
    IFS=':' read -r ac_file rest <<< "${row}"
    # opponent spec is everything between the first and last colon
    opp_spec="${rest%:*}"
    label="${rest##*:}"

    printf "%-60s  " "${label}"
    if [ ! -f "${ac_file}" ]; then
        printf "(missing AC file)\n"; continue
    fi
    out=$(${PY} run.py benchmark \
        --black "ac:agent_file=${ac_file}" \
        --white "${opp_spec}" \
        --board-size 9 -n "${N_GAMES}" 2>/dev/null)
    wins=$(echo "${out}" | grep -oE "ac:agent_file=[^[:space:]]+ wins: +[0-9]+" \
            | head -1 | grep -oE "[0-9]+$")
    if [ -z "${wins}" ]; then
        printf "err\n"
    else
        pct=$(( 100 * wins / N_GAMES ))
        printf "%3d%%  (%s wins)\n" "${pct}" "${wins}"
    fi
done
