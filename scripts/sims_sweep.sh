#!/bin/bash
# Sims sweep: same agent at varied num_simulations vs fixed opponent.
#
# Headline figure for the thesis: at insufficient value-head quality,
# more MCTS search amplifies noise and hurts play. Two anecdotal data
# points (sims=25 → 13%, sims=100 → 3% on r0_value_trained) suggested
# a monotonic-degradation curve; this sweep makes it a curve.
#
# Per cell N=50 games, color-swapped. Wilson 95% CI ~±14pp at 50% wins.
# Total expected wall ~2.5h at the default sims list.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

AGENT=${AGENT:-runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints/r0_value_trained.h5}
OPP=${OPP:-gnugo:level=1}
N_GAMES=${N_GAMES:-50}
SIMS_LIST=${SIMS_LIST:-"1 5 10 25 50 100 200"}
OUT_CSV=${OUT_CSV:-results/sims_sweep_r0_value_trained.csv}
BOARD_SIZE=${BOARD_SIZE:-9}

if [ ! -f "${AGENT}" ]; then
    echo "ERROR: agent not found: ${AGENT}"
    exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"
mkdir -p logs
echo "sims,n_games,wins,win_pct,wilson_ci_low,wilson_ci_high,wall_seconds" > "${OUT_CSV}"

echo "===== Sims sweep ====="
echo "  agent: ${AGENT}"
echo "  opponent: ${OPP}"
echo "  N per cell: ${N_GAMES}"
echo "  sims list: ${SIMS_LIST}"
echo "  output: ${OUT_CSV}"
echo

printf "%-6s  %-10s  %-7s  %-15s  %s\n" "sims" "wins/N" "win%" "Wilson 95% CI" "wall"
printf '=%.0s' {1..60}; printf "\n"

for sims in ${SIMS_LIST}; do
    t0=$(date +%s)
    raw_log="logs/sims_sweep_sims${sims}.log"
    ${PY} run.py benchmark \
        --black "alphazero:agent_file=${AGENT},num_simulations=${sims},temperature=0.0" \
        --white "${OPP}" \
        --board-size "${BOARD_SIZE}" -n "${N_GAMES}" > "${raw_log}" 2>&1
    rc=$?
    t1=$(date +%s)
    wall=$((t1 - t0))
    wins=$(grep -oE "wins: +[0-9]+" "${raw_log}" | head -1 | grep -oE "[0-9]+$")
    wins=${wins:-0}

    # Compute win pct + Wilson CI in Python for accuracy.
    read pct ci_lo ci_hi < <(${PY} -c "
from alphigo.benchmark import wilson_ci
n=${N_GAMES}; w=${wins}
pct = 100.0 * w / n if n else 0.0
lo, hi = wilson_ci(w, n)
print(f'{pct:.1f} {lo*100:.1f} {hi*100:.1f}')
")
    printf "%-6s  %-10s  %-7s  [%5s, %5s]  %ss  (rc=%d)\n" \
        "${sims}" "${wins}/${N_GAMES}" "${pct}%" "${ci_lo}%" "${ci_hi}%" "${wall}" "${rc}"
    echo "${sims},${N_GAMES},${wins},${pct},${ci_lo},${ci_hi},${wall}" >> "${OUT_CSV}"
done

echo
printf '=%.0s' {1..60}; printf "\n"
echo "===== sweep done — full CSV: ${OUT_CSV} ====="
cat "${OUT_CSV}"
