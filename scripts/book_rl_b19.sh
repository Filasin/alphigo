#!/bin/bash
# Book-faithful 19x19 PG/Q/AC runs + 2 targeted ablations.
#
# Protocol (uniform):
#   - 300 self-play games per round
#   - 100-game color-swapped paired eval challenger vs reference per round
#   - Promote on >= 54% (matches book's 262/480 threshold)
#   - Cap: 5 promotions OR 48h wall time per run, whichever first
#   - Hard ceiling of 60 rounds so an unlucky cold-start can't spin forever
#   - Early-stop: 15 consecutive rounds without a promotion
#
# Anchor runs (book-faithful):
#   A1 PG   elevenplane / large  / SGD lr=0.02  / cold start
#   A2 Q    elevenplane / large  / lr=0.01      / cold start
#   A3 AC   elevenplane / large  / lr=0.01      / cold start
#
# Ablations:
#   B1 PG   elevenplane / large  / SGD lr=0.02  / bootstrap from
#           supervised_elevenplane_large_adam_b19 (best epoch=33, val_acc=0.33)
#   B2 Q    oneplane / book_2layer / lr=1e-1    / 9x9-validated config
#
# Launch a single method with:
#   METHOD=A1 nohup bash scripts/book_rl_b19.sh > logs/book_rl_A1.log 2>&1 &
# or run them sequentially with:
#   nohup bash scripts/book_rl_b19.sh > logs/book_rl_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

# Shared protocol knobs.
BOARD_SIZE=19
GAMES=300
EVAL_GAMES=100
MIN_WIN=0.54
MAX_PROMOTIONS=5
MAX_WALL_H=48
EARLY_STOP=15
ROUNDS=60   # hard ceiling; real stop is on promotions/wall-time/early-stop

BOOTSTRAP_PG=checkpoints/supervised_elevenplane_large_adam_b19_model_epoch_33.h5

mkdir -p logs

run_A1() {
    echo "===== A1: PG elevenplane/large cold (book) ====="
    ${PY} -u train_rl.py \
        --method pg --board-size ${BOARD_SIZE} \
        --encoder elevenplane --network large \
        --lr 0.02 --bs 512 \
        --games ${GAMES} --rounds ${ROUNDS} \
        --eval-games ${EVAL_GAMES} --min-win-rate ${MIN_WIN} \
        --max-promotions ${MAX_PROMOTIONS} --max-wall-hours ${MAX_WALL_H} \
        --early-stop-rounds ${EARLY_STOP} \
        --seed 1
}

run_A2() {
    echo "===== A2: Q elevenplane/large cold (book) ====="
    ${PY} -u train_rl.py \
        --method q --board-size ${BOARD_SIZE} \
        --encoder elevenplane --network large \
        --lr 0.01 --bs 512 \
        --games ${GAMES} --rounds ${ROUNDS} \
        --eval-games ${EVAL_GAMES} --min-win-rate ${MIN_WIN} \
        --max-promotions ${MAX_PROMOTIONS} --max-wall-hours ${MAX_WALL_H} \
        --early-stop-rounds ${EARLY_STOP} \
        --seed 1
}

run_A3() {
    echo "===== A3: AC elevenplane/large cold (book) ====="
    ${PY} -u train_rl.py \
        --method ac --board-size ${BOARD_SIZE} \
        --encoder elevenplane --network large \
        --lr 0.01 --bs 512 \
        --games ${GAMES} --rounds ${ROUNDS} \
        --eval-games ${EVAL_GAMES} --min-win-rate ${MIN_WIN} \
        --max-promotions ${MAX_PROMOTIONS} --max-wall-hours ${MAX_WALL_H} \
        --early-stop-rounds ${EARLY_STOP} \
        --seed 1
}

run_B1() {
    echo "===== B1: PG elevenplane/large bootstrapped from supervised ====="
    if [ ! -f "${BOOTSTRAP_PG}" ]; then
        echo "ERROR: bootstrap not found: ${BOOTSTRAP_PG}"
        return 1
    fi
    ${PY} -u train_rl.py \
        --method pg --board-size ${BOARD_SIZE} \
        --encoder elevenplane --network large \
        --bootstrap "${BOOTSTRAP_PG}" \
        --lr 0.02 --bs 512 \
        --games ${GAMES} --rounds ${ROUNDS} \
        --eval-games ${EVAL_GAMES} --min-win-rate ${MIN_WIN} \
        --max-promotions ${MAX_PROMOTIONS} --max-wall-hours ${MAX_WALL_H} \
        --early-stop-rounds ${EARLY_STOP} \
        --seed 2
}

run_B2() {
    echo "===== B2: Q oneplane/book_2layer (9x9-validated config) ====="
    ${PY} -u train_rl.py \
        --method q --board-size ${BOARD_SIZE} \
        --encoder oneplane --network book_2layer \
        --lr 0.1 --bs 512 \
        --games ${GAMES} --rounds ${ROUNDS} \
        --eval-games ${EVAL_GAMES} --min-win-rate ${MIN_WIN} \
        --max-promotions ${MAX_PROMOTIONS} --max-wall-hours ${MAX_WALL_H} \
        --early-stop-rounds ${EARLY_STOP} \
        --seed 1
}

case "${METHOD:-all}" in
    A1) run_A1 ;;
    A2) run_A2 ;;
    A3) run_A3 ;;
    B1) run_B1 ;;
    B2) run_B2 ;;
    all)
        run_A1
        run_A2
        run_A3
        run_B1
        run_B2
        ;;
    *)
        echo "Unknown METHOD='${METHOD}'. Use A1|A2|A3|B1|B2|all."
        exit 2
        ;;
esac

echo "===== book_rl_b19.sh done at $(date) ====="
