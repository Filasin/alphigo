#!/bin/bash
# Bench every round of every q-dqn-sweep combo vs Random. N=100 default.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

ENCODERS=(oneplane sevenplane alphazero)

printf "combo                          "
for r in 000 001 002 003 004 005; do printf "r%s   " "$r"; done
printf "\n"; printf "%.s-" {1..80}; printf "\n"

for enc in "${ENCODERS[@]}"; do
    dir="rl_agents/9x9_${enc}_small_q_dqn"
    printf "%-30s  " "${enc}+small"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; continue; fi
    for r in 000 001 002 003 004 005; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black q-dqn:agent_file="${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
done
