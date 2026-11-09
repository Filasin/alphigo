#!/bin/bash
# Bench all lr-sensitivity variants (Q + PG, oneplane headline) vs
# Random at N=100 per checkpoint. Handles both lr-tagged dirs and
# the canonical name.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

bench_combo() {
    local label="$1" dir="$2" agent_type="$3"
    printf "%-32s  " "${label}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; return; fi
    for r in 000 001 002 003 004 005; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black "${agent_type}:agent_file=${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
}

printf "lr sensitivity (oneplane headline)   "
for r in 000 001 002 003 004 005; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..90}; printf "\n"

# Q oneplane + book_2layer at varying lr
bench_combo "Q   lr=1e-4 (one+book2)"  "rl_agents/9x9_oneplane_book_2layer_q_lr1e-4"  q
bench_combo "Q   lr=1e-3 (one+book2)"  "rl_agents/9x9_oneplane_book_2layer_q_lr1e-3"  q
bench_combo "Q   lr=1e-2 (one+book2)"  "rl_agents/9x9_oneplane_book_2layer_q_lr1e-2"  q
bench_combo "Q   lr=0.1  (one+book2)"  "rl_agents/9x9_oneplane_book_2layer_q_lr0.1"   q

# PG oneplane + small at varying lr
bench_combo "PG  lr=1e-4 (one+small)"  "rl_agents/9x9_oneplane_small_pg_lr1e4"        pg
bench_combo "PG  lr=1e-3 (one+small)"  "rl_agents/9x9_oneplane_small_pg_lr1e-3"       pg
bench_combo "PG  lr=1e-2 (one+small)"  "rl_agents/9x9_oneplane_small_pg_lr1e-2"       pg
bench_combo "PG  lr=0.1  (one+small)"  "rl_agents/9x9_oneplane_small_pg_lr0.1"        pg
