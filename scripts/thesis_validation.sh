#!/bin/bash
# Thesis-validation runs to upgrade Claim 3 ("simpler wins") from a
# hypothesis to a multi-seed, full-grid, long-budget result.
#
#   E1. Complete the oneplane row: oneplane x {medium_nobn, large_nobn,
#       alphazero_nobn} at the same 5x200 budget.
#   E2. Multi-seed (3 seeds) for the 3 headline combos
#       (oneplane/sevenplane/alphazero + book_2layer).
#   E3. Extend alphazero+book_2layer from 10 -> 30 rounds.
#
# Existing single-seed runs are reused as "seed 1":
#   oneplane+book_2layer  -> rl_agents/9x9_oneplane_book_2layer_q_lr0.1
#   sevenplane+book_2layer -> rl_agents/9x9_sevenplane_book_2layer_q (canonical)
#   alphazero+book_2layer  -> rl_agents/9x9_alphazero_book_2layer_q (canonical, 10r)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

train_q() {
    local enc="$1" ext="$2"
    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method q \
        --rounds 5 \
        --games 200 \
        --epsilon 0.1 \
        --lr 0.1 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
}

# ───────── E1: oneplane x extractors ─────────
echo "==================== E1: oneplane row completion ===================="
for ext in medium_nobn large_nobn alphazero_nobn; do
    echo "--- oneplane + ${ext} ---"
    rm -rf "rl_agents/9x9_oneplane_${ext}_q" "rl_experience/9x9_oneplane_${ext}_q"
    train_q oneplane "${ext}"
done

# ───────── E2: multi-seed (rename existing as seed1, train seeds 2 & 3) ─────────
echo "==================== E2: multi-seed retrains ===================="
# Promote existing canonicals to _seed1 (oneplane already at _lr0.1; just copy-link)
if [ -d "rl_agents/9x9_oneplane_book_2layer_q_lr0.1" ] && \
   [ ! -d "rl_agents/9x9_oneplane_book_2layer_q_seed1" ]; then
    cp -r "rl_agents/9x9_oneplane_book_2layer_q_lr0.1" \
          "rl_agents/9x9_oneplane_book_2layer_q_seed1"
fi
for combo in "sevenplane:book_2layer" "alphazero:book_2layer"; do
    enc="${combo%:*}"; ext="${combo#*:}"
    src="rl_agents/9x9_${enc}_${ext}_q"
    dst="rl_agents/9x9_${enc}_${ext}_q_seed1"
    if [ -d "${src}" ] && [ ! -d "${dst}" ]; then mv "${src}" "${dst}"; fi
    src_exp="rl_experience/9x9_${enc}_${ext}_q"
    dst_exp="rl_experience/9x9_${enc}_${ext}_q_seed1"
    if [ -d "${src_exp}" ] && [ ! -d "${dst_exp}" ]; then mv "${src_exp}" "${dst_exp}"; fi
done

for seed in 2 3; do
    for combo in "oneplane:book_2layer" "sevenplane:book_2layer" "alphazero:book_2layer"; do
        enc="${combo%:*}"; ext="${combo#*:}"
        echo "--- ${enc}+${ext} seed ${seed} ---"
        rm -rf "rl_agents/9x9_${enc}_${ext}_q" "rl_experience/9x9_${enc}_${ext}_q"
        train_q "${enc}" "${ext}"
        [ -d "rl_agents/9x9_${enc}_${ext}_q" ] && \
            mv "rl_agents/9x9_${enc}_${ext}_q" "rl_agents/9x9_${enc}_${ext}_q_seed${seed}"
        [ -d "rl_experience/9x9_${enc}_${ext}_q" ] && \
            mv "rl_experience/9x9_${enc}_${ext}_q" "rl_experience/9x9_${enc}_${ext}_q_seed${seed}"
    done
done

# ───────── E3: extend alphazero+book_2layer to 30 rounds ─────────
echo "==================== E3: alphazero long-budget extension ===================="
# Restore canonical from _seed1 so resume mode picks up at round 10
if [ ! -d "rl_agents/9x9_alphazero_book_2layer_q" ] && \
   [ -d "rl_agents/9x9_alphazero_book_2layer_q_seed1" ]; then
    cp -r "rl_agents/9x9_alphazero_book_2layer_q_seed1" \
          "rl_agents/9x9_alphazero_book_2layer_q"
fi
if [ ! -d "rl_experience/9x9_alphazero_book_2layer_q" ] && \
   [ -d "rl_experience/9x9_alphazero_book_2layer_q_seed1" ]; then
    cp -r "rl_experience/9x9_alphazero_book_2layer_q_seed1" \
          "rl_experience/9x9_alphazero_book_2layer_q"
fi

${PY} train_rl.py \
    --board-size 9 \
    --encoder alphazero \
    --network book_2layer \
    --method q \
    --rounds 30 \
    --games 200 \
    --epsilon 0.1 \
    --lr 0.1 2>&1 | \
    grep -E 'init|ROUND|Round [0-9]+ done|Traceback|Error|FAIL|Killed'

# Tag the long-budget run so it isn't accidentally clobbered
mv "rl_agents/9x9_alphazero_book_2layer_q" "rl_agents/9x9_alphazero_book_2layer_q_long30"
[ -d "rl_experience/9x9_alphazero_book_2layer_q" ] && \
    mv "rl_experience/9x9_alphazero_book_2layer_q" "rl_experience/9x9_alphazero_book_2layer_q_long30"

echo "==================== thesis validation training done ===================="
