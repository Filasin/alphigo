#!/bin/bash
# Benchmark PG agents vs Random at the same evaluation protocol as the
# Q sweep (N=100 games per checkpoint by default). Reports both budget-
# matched (round 5, comparable to Q's 5-round sweep) and asymptotic
# (round 100) win-rates per combo.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

COMBOS=(
    "oneplane:small"
    "sevenplane:large"
    "alphazero:alphazero"
    "elevenplane:large"
)

# Rounds to bench: budget-matched (0..5) and asymptotic (25, 50, 100).
ROUNDS=(000 001 002 003 004 005 025 050 100)

printf "combo                          "
for r in "${ROUNDS[@]}"; do printf "r%s   " "$r"; done
printf "\n"
printf "%.s-" {1..120}; printf "\n"

for combo in "${COMBOS[@]}"; do
    enc="${combo%:*}"; net="${combo#*:}"
    dir="rl_agents/9x9_${enc}_${net}"
    printf "%-30s  " "${enc}+${net}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; continue; fi
    for r in "${ROUNDS[@]}"; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black pg:agent_file="${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
done
