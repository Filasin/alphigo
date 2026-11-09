#!/bin/bash
# Direct head-to-head matches between the best-trained agents per method.
# 100 games per matchup, with auto color swap.
#
# Best-of agents (5-round budget, lr=0.1 unless noted):
#   Q       : oneplane + book_2layer  (round 5)  — 72% vs Random
#   Q-DQN   : best combo from q_dqn_sweep, round 5
#   PG-1e-4 : oneplane + small        (round 5 of legacy lr=1e-4 backup)
#   PG-0.1  : oneplane + small        (round 5 of fresh lr=0.1)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

Q_BEST="rl_agents/9x9_oneplane_book_2layer_q_lr0.1/round_005.h5"
# fall back to canonical if the lr=0.1 rename hasn't happened yet
if [ ! -f "${Q_BEST}" ]; then
    Q_BEST="rl_agents/9x9_oneplane_book_2layer_q/round_005.h5"
fi
PG_LR1E4_BEST="rl_agents/9x9_oneplane_small_pg_lr1e4/round_005.h5"
PG_LR01_BEST="rl_agents/9x9_oneplane_small_pg_lr0.1/round_005.h5"
# Q-DQN: pick the strongest combo at round 5 — by default oneplane+book_2layer
# (will work even if other combos turn out stronger; user can re-run with edits)
QDQN_BEST="rl_agents/9x9_oneplane_small_q_dqn/round_005.h5"

run_match() {
    local label="$1" black="$2" white="$3"
    printf "%-50s  " "${label}"
    if [ ! -f "${black#*=}" ] || [ ! -f "${white#*=}" ]; then
        printf "(missing checkpoint)\n"; return
    fi
    out=$(${PY} run.py benchmark --black "${black}" --white "${white}" \
        --board-size 9 -n "${N_GAMES}" 2>/dev/null)
    wins=$(echo "${out}" | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
    if [ -z "${wins}" ]; then printf "err\n"
    else pct=$(( 100 * wins / N_GAMES )); printf "%3d%% black wins\n" "${pct}"; fi
}

echo "=== Head-to-head matrix (N=${N_GAMES} per matchup, color-swapped) ==="
run_match "Q (5r)        vs PG-lr1e-4 (5r)" "q:agent_file=${Q_BEST}"      "pg:agent_file=${PG_LR1E4_BEST}"
run_match "Q (5r)        vs PG-lr0.1  (5r)" "q:agent_file=${Q_BEST}"      "pg:agent_file=${PG_LR01_BEST}"
run_match "Q (5r)        vs Q-DQN     (5r)" "q:agent_file=${Q_BEST}"      "q-dqn:agent_file=${QDQN_BEST}"
run_match "Q-DQN (5r)    vs PG-lr1e-4 (5r)" "q-dqn:agent_file=${QDQN_BEST}" "pg:agent_file=${PG_LR1E4_BEST}"
run_match "Q-DQN (5r)    vs PG-lr0.1  (5r)" "q-dqn:agent_file=${QDQN_BEST}" "pg:agent_file=${PG_LR01_BEST}"
run_match "PG-lr1e-4 (5r) vs PG-lr0.1 (5r)" "pg:agent_file=${PG_LR1E4_BEST}" "pg:agent_file=${PG_LR01_BEST}"
echo "==="
