#!/bin/bash
# Multi-seed for the two new headline candidates uncovered in the
# first round of validation:
#   E4. oneplane + large_nobn  (5 rounds)   — beat book_2layer 80% vs 66%
#   E5. alphazero + book_2layer (30 rounds) — caught up at 6x budget
# Seeds 1 are the existing single-seed runs from thesis_validation.sh.
# Trains seeds 2 and 3 here.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

train_q() {
    local enc="$1" ext="$2" rounds="$3"
    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method q \
        --rounds "${rounds}" \
        --games 200 \
        --epsilon 0.1 \
        --lr 0.1 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
}

# ───────── E4: multi-seed oneplane + large_nobn (5 rounds) ─────────
echo "==================== E4: oneplane+large_nobn multi-seed ===================="
# Promote existing run to seed1
src="rl_agents/9x9_oneplane_large_nobn_q"
dst="rl_agents/9x9_oneplane_large_nobn_q_seed1"
if [ -d "${src}" ] && [ ! -d "${dst}" ]; then mv "${src}" "${dst}"; fi
src_exp="rl_experience/9x9_oneplane_large_nobn_q"
dst_exp="rl_experience/9x9_oneplane_large_nobn_q_seed1"
if [ -d "${src_exp}" ] && [ ! -d "${dst_exp}" ]; then mv "${src_exp}" "${dst_exp}"; fi

for seed in 2 3; do
    echo "--- oneplane+large_nobn seed ${seed} ---"
    rm -rf "rl_agents/9x9_oneplane_large_nobn_q" \
           "rl_experience/9x9_oneplane_large_nobn_q"
    train_q oneplane large_nobn 5
    [ -d "rl_agents/9x9_oneplane_large_nobn_q" ] && \
        mv "rl_agents/9x9_oneplane_large_nobn_q" \
           "rl_agents/9x9_oneplane_large_nobn_q_seed${seed}"
    [ -d "rl_experience/9x9_oneplane_large_nobn_q" ] && \
        mv "rl_experience/9x9_oneplane_large_nobn_q" \
           "rl_experience/9x9_oneplane_large_nobn_q_seed${seed}"
done

# ───────── E5: multi-seed alphazero+book_2layer long (30 rounds) ─────────
echo "==================== E5: alphazero+book_2layer long-budget multi-seed ===================="
# Existing _long30 dir is treated as seed1
src="rl_agents/9x9_alphazero_book_2layer_q_long30"
dst="rl_agents/9x9_alphazero_book_2layer_q_long30_seed1"
if [ -d "${src}" ] && [ ! -d "${dst}" ]; then mv "${src}" "${dst}"; fi
src_exp="rl_experience/9x9_alphazero_book_2layer_q_long30"
dst_exp="rl_experience/9x9_alphazero_book_2layer_q_long30_seed1"
if [ -d "${src_exp}" ] && [ ! -d "${dst_exp}" ]; then mv "${src_exp}" "${dst_exp}"; fi

for seed in 2 3; do
    echo "--- alphazero+book_2layer 30r seed ${seed} ---"
    rm -rf "rl_agents/9x9_alphazero_book_2layer_q" \
           "rl_experience/9x9_alphazero_book_2layer_q"
    train_q alphazero book_2layer 30
    [ -d "rl_agents/9x9_alphazero_book_2layer_q" ] && \
        mv "rl_agents/9x9_alphazero_book_2layer_q" \
           "rl_agents/9x9_alphazero_book_2layer_q_long30_seed${seed}"
    [ -d "rl_experience/9x9_alphazero_book_2layer_q" ] && \
        mv "rl_experience/9x9_alphazero_book_2layer_q" \
           "rl_experience/9x9_alphazero_book_2layer_q_long30_seed${seed}"
done

echo "==================== thesis validation v2 done ===================="
