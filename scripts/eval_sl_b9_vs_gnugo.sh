#!/bin/bash
# Evaluate every 9x9 SL epoch_100 checkpoint against GnuGo level=1.
#
# We have 13 different SL configurations under checkpoints/9x9_*_b9_*.h5
# but never measured their absolute strength vs an external baseline.
# This script fixes that by running 30 color-swapped games per checkpoint
# vs gnugo:level=1, saving the win-rate matrix to JSON.
#
# Special handling for the alphazero/alphazero checkpoint: it's a
# two-head network, so we eval it as a ZeroAgent at sims=25 (matching
# AZ training-time strength) rather than as a single-head DL agent.
# This is the candidate for AZ bootstrap.
#
# Time estimate: ~24 min total (12 × ~1.5min + 1 × ~6min).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

N_GAMES=${N_GAMES:-30}
OPP=${OPP:-gnugo:level=1}
EPOCH=${EPOCH:-100}
OUT=${OUT:-results/sl_b9_vs_gnugo.json}

# Lines: <agent_spec>:<label>
# DL spec for single-head policy nets; alphazero spec for two-head.
ENTRIES=(
    "dl:agent_file=checkpoints/9x9_oneplane_small_adam_b9_model_epoch_${EPOCH}.h5|oneplane/small"
    "dl:agent_file=checkpoints/9x9_oneplane_medium_adam_b9_model_epoch_${EPOCH}.h5|oneplane/medium"
    "dl:agent_file=checkpoints/9x9_oneplane_large_adam_b9_model_epoch_${EPOCH}.h5|oneplane/large"
    "dl:agent_file=checkpoints/9x9_sevenplane_small_adam_b9_model_epoch_${EPOCH}.h5|sevenplane/small"
    "dl:agent_file=checkpoints/9x9_sevenplane_medium_adam_b9_model_epoch_${EPOCH}.h5|sevenplane/medium"
    "dl:agent_file=checkpoints/9x9_sevenplane_large_adam_b9_model_epoch_${EPOCH}.h5|sevenplane/large"
    "dl:agent_file=checkpoints/9x9_elevenplane_small_adam_b9_model_epoch_${EPOCH}.h5|elevenplane/small"
    "dl:agent_file=checkpoints/9x9_elevenplane_medium_adam_b9_model_epoch_${EPOCH}.h5|elevenplane/medium"
    "dl:agent_file=checkpoints/9x9_elevenplane_large_adam_b9_model_epoch_${EPOCH}.h5|elevenplane/large"
    "dl:agent_file=checkpoints/9x9_agplane_small_adam_b9_model_epoch_${EPOCH}.h5|agplane/small"
    "dl:agent_file=checkpoints/9x9_agplane_medium_adam_b9_model_epoch_${EPOCH}.h5|agplane/medium"
    "dl:agent_file=checkpoints/9x9_agplane_large_adam_b9_model_epoch_${EPOCH}.h5|agplane/large"
    "dl:agent_file=checkpoints/9x9_alphazero_alphazero_adam_b9_model_epoch_${EPOCH}.h5|alphazero/alphazero (pure policy)"
)

mkdir -p "$(dirname "${OUT}")"
echo "{\"opponent\": \"${OPP}\", \"num_games\": ${N_GAMES}, \"epoch\": ${EPOCH}, \"results\": [" > "${OUT}"
first=1

printf "\n%-40s  %-10s  %s\n" "config" "wins/N" "wall"
printf '=%.0s' {1..70}; printf "\n"

for entry in "${ENTRIES[@]}"; do
    spec="${entry%%|*}"
    label="${entry#*|}"

    # Extract path from spec for existence check
    path=$(echo "${spec}" | grep -oE 'agent_file=[^,]+' | sed 's/agent_file=//')
    if [ ! -f "${path}" ]; then
        printf "%-40s  %-10s  %s\n" "${label}" "MISSING" "${path}"
        continue
    fi

    t0=$(date +%s)
    # Save raw benchmark output for forensic debugging if extraction fails.
    raw_log="logs/sl_b9_raw_$(echo "${label}" | tr '/ ()=,' '_').log"
    ${PY} run.py benchmark \
        --black "${spec}" --white "${OPP}" \
        --board-size 9 -n "${N_GAMES}" > "${raw_log}" 2>&1
    rc=$?
    t1=$(date +%s)
    # Wins line is ~4 lines below "BENCHMARK RESULTS", first occurrence is
    # our agent (black/A). Use a wide context window so we don't miss it.
    wins=$(grep -oE "wins: +[0-9]+" "${raw_log}" | head -1 | grep -oE "[0-9]+$")
    wall=$((t1 - t0))
    json_wins="${wins:-null}"
    if [ -z "${wins}" ]; then
        # Print last 5 lines so the error is visible inline.
        printf "%-40s  %-10s  rc=%d %ss (see %s)\n" "${label}" "?/${N_GAMES}" "${rc}" "${wall}" "${raw_log}"
    else
        printf "%-40s  %-10s  %ss\n" "${label}" "${wins}/${N_GAMES}" "${wall}"
    fi
    if [ ${first} -eq 0 ]; then echo "  ," >> "${OUT}"; fi
    first=0
    echo "  {\"label\": \"${label}\", \"spec\": \"${spec}\", \"wins\": ${json_wins}, \"num_games\": ${N_GAMES}, \"wall_seconds\": ${wall}, \"rc\": ${rc}, \"raw_log\": \"${raw_log}\"}" >> "${OUT}"
done

echo "]}" >> "${OUT}"
echo
printf '=%.0s' {1..70}; printf "\n"
echo "##################### SL eval done #####################"
echo "  saved to ${OUT}"
