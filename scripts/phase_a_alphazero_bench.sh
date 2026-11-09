#!/bin/bash
# Phase A AlphaZero bench: vs random N=100 at fine rounds + vs gnugo (level 0)
# N=20 at coarse rounds.  Output to logs/phase_a_alphazero_bench.jsonl for the
# figure / Elo-table generation script.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

OUT_JSONL="logs/phase_a_alphazero_bench.jsonl"
N_RANDOM=${N_RANDOM:-100}
N_GNUGO=${N_GNUGO:-20}
MCTS_SIMS=${MCTS_SIMS:-25}
ROUNDS_FINE="000 005 010 015 020 025 030 035 040 045 050"
ROUNDS_COARSE="000 010 025 050"

# Phase A AZ runs share the canonical Run dir (slug has no seed suffix).
# Per-seed bench requires the upstream slug-with-seed work; for now we
# bench the single canonical dir.
ENTRIES=(
    "runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed1/checkpoints:AZ seed1"
    "runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints:AZ seed2"
    "runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed3/checkpoints:AZ seed3"
)

bench_one() {
    local agent_file="$1" opp="$2" n="$3"
    if [ ! -f "${agent_file}" ]; then echo ""; return; fi
    ${PY} run.py benchmark \
        --black "alphazero:agent_file=${agent_file},num_simulations=${MCTS_SIMS}" \
        --white "${opp}" \
        --board-size 9 -n "${n}" 2>/dev/null \
        | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+"
}

mkdir -p "$(dirname ${OUT_JSONL})"
> "${OUT_JSONL}"

echo "===== Phase A AlphaZero bench: vs Random (N=${N_RANDOM}) every 5 rounds ====="
printf "%-25s  " "label"
for r in ${ROUNDS_FINE}; do printf "r%s " "$r"; done
printf "\n"
printf "%.s-" {1..130}; printf "\n"
for entry in "${ENTRIES[@]}"; do
    dir="${entry%%:*}"
    label="${entry##*:}"
    printf "%-25s  " "${label}"
    if [ ! -d "${dir}" ]; then printf "(missing dir)\n"; continue; fi
    for r in ${ROUNDS_FINE}; do
        wins=$(bench_one "${dir}/round_${r}.h5" random "${N_RANDOM}")
        if [ -z "${wins}" ]; then
            printf " --  "
        else
            pct=$(( 100 * wins / N_RANDOM ))
            printf "%3d%% " "${pct}"
            echo "{\"label\":\"${label}\",\"method\":\"alphazero\",\"round\":${r},\"opponent\":\"random\",\"n\":${N_RANDOM},\"wins\":${wins},\"pct\":${pct}}" >> "${OUT_JSONL}"
        fi
    done
    printf "\n"
done
echo

echo "===== Phase A AlphaZero bench: vs gnugo level 0 (N=${N_GNUGO}) at coarse rounds ====="
printf "%-25s  " "label"
for r in ${ROUNDS_COARSE}; do printf "r%s " "$r"; done
printf "\n"
printf "%.s-" {1..80}; printf "\n"
for entry in "${ENTRIES[@]}"; do
    dir="${entry%%:*}"
    label="${entry##*:}"
    printf "%-25s  " "${label}"
    if [ ! -d "${dir}" ]; then printf "(missing dir)\n"; continue; fi
    for r in ${ROUNDS_COARSE}; do
        wins=$(bench_one "${dir}/round_${r}.h5" "gnugo:level=0" "${N_GNUGO}")
        if [ -z "${wins}" ]; then
            printf " --  "
        else
            pct=$(( 100 * wins / N_GNUGO ))
            printf "%3d%% " "${pct}"
            echo "{\"label\":\"${label}\",\"method\":\"alphazero\",\"round\":${r},\"opponent\":\"gnugo_l0\",\"n\":${N_GNUGO},\"wins\":${wins},\"pct\":${pct}}" >> "${OUT_JSONL}"
        fi
    done
    printf "\n"
done
