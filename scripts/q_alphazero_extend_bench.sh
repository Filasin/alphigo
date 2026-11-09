#!/bin/bash
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-20}

COMBOS=(
    "alphazero:book_2layer"
    "alphazero:large_nobn"
)

printf "combo                                "
for r in 000 001 002 003 004 005 006 007 008 009 010; do
    printf "r%s   " "$r"
done
printf "\n"
printf "%.s-" {1..120}; printf "\n"

for combo in "${COMBOS[@]}"; do
    enc="${combo%:*}"; ext="${combo#*:}"
    dir="rl_agents/9x9_${enc}_${ext}_q"
    printf "%-35s  " "${enc}+${ext}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; continue; fi
    for r in 000 001 002 003 004 005 006 007 008 009 010; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --  "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black q:agent_file="${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err  "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%  " "${pct}"; fi
    done
    printf "\n"
done
