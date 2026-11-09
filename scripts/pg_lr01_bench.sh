#!/bin/bash
# Benchmark the lr=0.1 PG sweep vs Random at N=100 per checkpoint.
# Same encoder x network combos as pg_lr01_sweep.sh, rounds 0..5.
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

printf "combo                          "
for r in 000 001 002 003 004 005; do printf "r%s   " "$r"; done
printf "\n"; printf "%.s-" {1..90}; printf "\n"

for combo in "${COMBOS[@]}"; do
    enc="${combo%:*}"; net="${combo#*:}"
    dir="rl_agents/9x9_${enc}_${net}"
    printf "%-30s  " "${enc}+${net}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; continue; fi
    for r in 000 001 002 003 004 005; do
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
