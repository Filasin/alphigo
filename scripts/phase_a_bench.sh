#!/bin/bash
# Phase A bench: vs Random N=100 every 5 rounds; vs mcts:num_rounds=100 N=20
# at peak rounds (0, 10, 25, 50).  Output to logs/phase_a_bench.jsonl for
# the figure generation script.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

OUT_JSONL="logs/phase_a_bench.jsonl"
N_RANDOM=${N_RANDOM:-100}
N_MCTS=${N_MCTS:-20}
ROUNDS_FINE="000 005 010 015 020 025 030 035 040 045 050"
ROUNDS_COARSE="000 010 025 050"

# ─── Configurations to bench (matches phase_a_train.sh output dirs) ───
# Format: <method>:<dir>:<label>
ENTRIES=(
    "q:rl_agents/9x9_oneplane_book_2layer_q_big50x1000_seed1:Q seed1"
    "q:rl_agents/9x9_oneplane_book_2layer_q_big50x1000_seed2:Q seed2"
    "q:rl_agents/9x9_oneplane_book_2layer_q_big50x1000_seed3:Q seed3"
    "pg:rl_agents/9x9_oneplane_small_big50x1000_seed1:PG seed1"
    "pg:rl_agents/9x9_oneplane_small_big50x1000_seed2:PG seed2"
    "pg:rl_agents/9x9_oneplane_small_big50x1000_seed3:PG seed3"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_big50x1000_seed1:AC seed1"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_big50x1000_seed2:AC seed2"
    "ac:rl_agents/9x9_oneplane_book_2layer_ac_big50x1000_seed3:AC seed3"
)

bench_one() {
    local method="$1" agent_file="$2" opp="$3" n="$4"
    if [ ! -f "${agent_file}" ]; then echo "" ; return; fi
    ${PY} run.py benchmark \
        --black "${method}:agent_file=${agent_file}" --white "${opp}" \
        --board-size 9 -n "${n}" 2>/dev/null \
        | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+"
}

# Truncate output JSONL
> "${OUT_JSONL}"

echo "===== Phase A bench: vs Random (N=${N_RANDOM}) every 5 rounds ====="
printf "%-30s  " "label"
for r in ${ROUNDS_FINE}; do printf "r%s " "$r"; done
printf "\n"
printf "%.s-" {1..130}; printf "\n"
for entry in "${ENTRIES[@]}"; do
    method="${entry%%:*}"
    rest="${entry#*:}"
    dir="${rest%%:*}"
    label="${rest##*:}"
    printf "%-30s  " "${label}"
    if [ ! -d "${dir}" ]; then printf "(missing dir)\n"; continue; fi
    for r in ${ROUNDS_FINE}; do
        wins=$(bench_one "${method}" "${dir}/round_${r}.h5" random "${N_RANDOM}")
        if [ -z "${wins}" ]; then
            printf " --  "
        else
            pct=$(( 100 * wins / N_RANDOM ))
            printf "%3d%% " "${pct}"
            echo "{\"label\":\"${label}\",\"method\":\"${method}\",\"round\":${r},\"opponent\":\"random\",\"n\":${N_RANDOM},\"wins\":${wins},\"pct\":${pct}}" >> "${OUT_JSONL}"
        fi
    done
    printf "\n"
done
echo

echo "===== Phase A bench: vs mcts:num_rounds=100 (N=${N_MCTS}) at coarse rounds ====="
printf "%-30s  " "label"
for r in ${ROUNDS_COARSE}; do printf "r%s " "$r"; done
printf "\n"
printf "%.s-" {1..80}; printf "\n"
for entry in "${ENTRIES[@]}"; do
    method="${entry%%:*}"
    rest="${entry#*:}"
    dir="${rest%%:*}"
    label="${rest##*:}"
    printf "%-30s  " "${label}"
    if [ ! -d "${dir}" ]; then printf "(missing dir)\n"; continue; fi
    for r in ${ROUNDS_COARSE}; do
        wins=$(bench_one "${method}" "${dir}/round_${r}.h5" "mcts:num_rounds=100" "${N_MCTS}")
        if [ -z "${wins}" ]; then
            printf " --  "
        else
            pct=$(( 100 * wins / N_MCTS ))
            printf "%3d%% " "${pct}"
            echo "{\"label\":\"${label}\",\"method\":\"${method}\",\"round\":${r},\"opponent\":\"mcts100\",\"n\":${N_MCTS},\"wins\":${wins},\"pct\":${pct}}" >> "${OUT_JSONL}"
        fi
    done
    printf "\n"
done
