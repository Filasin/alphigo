#!/bin/bash
# MCTS-correctness sims sweep on 19x19.
#
# Goal: prove the MCTS implementation is sound by varying ONLY the value
# source while keeping policy, search, and opponent fixed.
#
# Three curves on the same axes:
#   - value_mode=network : the current behaviour (degrades with sims — F1)
#   - value_mode=zero    : V=0.0 always (MCTS = policy-prior-guided UCT)
#   - value_mode=rollout : V from a random rollout to game end (AlphaGo Lee style)
#
# Predictions:
#   - 'zero'    : roughly flat or mildly improving with sims
#                 → confirms MCTS itself doesn't degrade play
#   - 'rollout' : monotonically improving with sims
#                 → confirms MCTS works given a non-misleading V signal
#   - 'network' : reproduces the existing degradation curve
#
# Cost: rollout games are slower than network games; budget N=15 per cell.
#
# Wall: ~6h (3 modes × 4 sims × ~15 games × ~5min avg).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

AGENT=${AGENT:-runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_value_b19_allchunks.h5}
OPP=${OPP:-gnugo:level=5}
N_GAMES=${N_GAMES:-15}
SIMS_LIST=${SIMS_LIST:-"1 25 50 200"}
BOARD_SIZE=19
OUT_CSV=${OUT_CSV:-results/mcts_correctness_b19.csv}
ROLLOUT_DEPTH=${ROLLOUT_DEPTH:-80}

if [ ! -f "${AGENT}" ]; then
    echo "ERROR: agent not found: ${AGENT}"
    exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"
mkdir -p logs
echo "value_mode,sims,n_games,wins,win_pct,wilson_ci_low,wilson_ci_high,wall_seconds" > "${OUT_CSV}"

echo "===== MCTS correctness sweep ====="
echo "  agent: ${AGENT}"
echo "  opponent: ${OPP}"
echo "  N per cell: ${N_GAMES}"
echo "  sims list: ${SIMS_LIST}"
echo "  modes: network, zero, rollout (depth=${ROLLOUT_DEPTH})"
echo "  output: ${OUT_CSV}"
echo

printf "%-8s  %-6s  %-10s  %-7s  %-15s  %s\n" \
    "mode" "sims" "wins/N" "win%" "Wilson 95% CI" "wall"
printf '=%.0s' {1..70}; printf "\n"

for mode in network zero rollout; do
    for sims in ${SIMS_LIST}; do
        t0=$(date +%s)
        raw_log="logs/mcts_correctness_${mode}_sims${sims}.log"

        spec="alphazero:agent_file=${AGENT},num_simulations=${sims},temperature=0.0,value_mode=${mode}"
        if [ "${mode}" = "rollout" ]; then
            spec="${spec},rollout_depth=${ROLLOUT_DEPTH}"
        fi

        ${PY} run.py benchmark \
            --black "${spec}" \
            --white "${OPP}" \
            --board-size "${BOARD_SIZE}" -n "${N_GAMES}" \
            > "${raw_log}" 2>&1
        rc=$?
        t1=$(date +%s)
        wall=$((t1 - t0))
        wins=$(grep -oE "wins: +[0-9]+" "${raw_log}" | head -1 | grep -oE "[0-9]+$")
        wins=${wins:-0}

        read pct ci_lo ci_hi < <(${PY} -c "
from alphigo.benchmark import wilson_ci
n=${N_GAMES}; w=${wins}
pct = 100.0 * w / n if n else 0.0
lo, hi = wilson_ci(w, n)
print(f'{pct:.1f} {lo*100:.1f} {hi*100:.1f}')
")
        printf "%-8s  %-6s  %-10s  %-7s  [%5s, %5s]  %ss  (rc=%d)\n" \
            "${mode}" "${sims}" "${wins}/${N_GAMES}" "${pct}%" \
            "${ci_lo}%" "${ci_hi}%" "${wall}" "${rc}"
        echo "${mode},${sims},${N_GAMES},${wins},${pct},${ci_lo},${ci_hi},${wall}" >> "${OUT_CSV}"
    done
done

echo
printf '=%.0s' {1..70}; printf "\n"
echo "===== sweep done — full CSV: ${OUT_CSV} ====="
cat "${OUT_CSV}"
