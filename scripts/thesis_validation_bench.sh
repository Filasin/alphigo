#!/bin/bash
# Bench all thesis-validation runs vs Random at N=100.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

bench_dir() {
    local label="$1" dir="$2" rounds="$3"
    printf "%-46s  " "${label}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; return; fi
    for r in ${rounds}; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black q:agent_file="${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
}

# ───────── E1: oneplane row at 5 rounds ─────────
echo "===== E1: oneplane row (5 rounds) ====="
printf "%-46s  " "combo"
for r in 000 001 002 003 004 005; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..120}; printf "\n"

ROUNDS_5="000 001 002 003 004 005"
bench_dir "oneplane + book_2layer (existing seed1)" \
          "rl_agents/9x9_oneplane_book_2layer_q_lr0.1" "${ROUNDS_5}"
bench_dir "oneplane + medium_nobn" \
          "rl_agents/9x9_oneplane_medium_nobn_q"      "${ROUNDS_5}"
bench_dir "oneplane + large_nobn" \
          "rl_agents/9x9_oneplane_large_nobn_q"       "${ROUNDS_5}"
bench_dir "oneplane + alphazero_nobn" \
          "rl_agents/9x9_oneplane_alphazero_nobn_q"   "${ROUNDS_5}"
echo

# ───────── E2: multi-seed at 5 rounds ─────────
echo "===== E2: multi-seed (5 rounds, 3 seeds each) ====="
printf "%-46s  " "combo (seed)"
for r in 000 001 002 003 004 005; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..120}; printf "\n"

for combo in "oneplane:book_2layer" "sevenplane:book_2layer" "alphazero:book_2layer"; do
    enc="${combo%:*}"; ext="${combo#*:}"
    for seed in 1 2 3; do
        bench_dir "${enc}+${ext} (seed ${seed})" \
                  "rl_agents/9x9_${enc}_${ext}_q_seed${seed}" "${ROUNDS_5}"
    done
done
echo

# ───────── E3: alphazero long (30 rounds) ─────────
echo "===== E3: alphazero+book_2layer long-budget extension ====="
ROUNDS_30="000 002 005 007 010 012 015 017 020 022 025 027 030"
printf "%-46s  " "combo"
for r in ${ROUNDS_30}; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..160}; printf "\n"
bench_dir "alphazero+book_2layer (long, every ~3r)" \
          "rl_agents/9x9_alphazero_book_2layer_q_long30" "${ROUNDS_30}"
