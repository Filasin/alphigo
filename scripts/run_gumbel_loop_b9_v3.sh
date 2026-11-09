#!/bin/bash
# Gumbel AlphaZero self-play loop on 9x9 — v3 with pass action.
# Uses 2-head 8b64f architecture, policy output = 82 (81 board + pass).
# Adam optimizer:
#   - Learning rate: 3e-4 for rounds 1-14, 1e-4 for rounds 15+
#   - Training window: growing, capped at 20 rounds
#   - 2000 games per round, batch_mcts=32, 2 epochs per round
#   - Eval (GNU Go + probe) every round; rolling h2h every 10
#   - Total rounds: 40
#
# Usage:
#   bash scripts/run_gumbel_loop_b9_v3.sh [--rounds N] [--games N]
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config ───────────────────────────────────────────────────────────
BOARD=9
BLOCKS=8
FILTERS=64
SIMS=32
GUMBEL_M=16
GAMES_PER_ROUND=2000
WORKERS=8
BATCH_MCTS=32
TRAIN_EPOCHS=1
TRAIN_BS=1024
ROUNDS=40
MAX_WINDOW=20
EVAL_EVERY=1
H2H_GAMES=20
GNUGO_GAMES=20
SEED_AGENT=runs/bootstrap_b9_v3/model_bootstrap_agent.h5
BOOTSTRAP_GLOB_DIR=data/bootstrap_b9_h5

RUN_DIR=runs/gumbel_loop_b9_v3
DATA_DIR=data/gumbel_loop_b9_v3
RESULTS_CSV=results/gumbel_loop_b9_v3.csv
LOG=logs/gumbel_loop_b9_v3.log

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --games) GAMES_PER_ROUND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[gumbel-v3] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

CKPT_DIR="$RUN_DIR/checkpoints"
ROUND0_AGENT="$CKPT_DIR/round_000_agent.h5"

# ── CSV header ───────────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,h2h0_wins,h2h0_n,gnugo_wins,gnugo_n,policy_ce,value_acc,h2hprev_wins,h2hprev_n,positions,gen_secs,gnugo_ph_wins" > "$RESULTS_CSV"
fi

# ── LR schedule ─────────────────────────────────────────────────────
get_lr() {
    local r=$1
    if [ "$r" -lt 15 ]; then
        echo "3e-4"
    else
        echo "1e-4"
    fi
}

# ── Helpers ──────────────────────────────────────────────────────────
run_h2h () {
    local out
    out=$("$PY" -u scripts/head_to_head.py \
        --agent-a "$1" --agent-b "$2" \
        --num-games "$H2H_GAMES" --board-size "$BOARD" \
        --sims "$SIMS" --search gumbel --gumbel-m "$GUMBEL_M" \
        --seed "$3" 2>&1 | tee -a "$LOG")
    H2H_WINS=$(echo "$out" | sed -n 's/.*A wins: \([0-9][0-9]*\).*/\1/p')
}

run_gnugo () {
    local out
    out=$("$PY" -u scripts/quick_eval_mcts.py \
        --agent "$1" --sims "$SIMS" --games "$GNUGO_GAMES" \
        --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
        --opponent gnugo:level=1 2>&1 | tee -a "$LOG")
    GNUGO_WINS=$(echo "$out" | sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p')
}

run_gnugo_passhack () {
    local out
    out=$("$PY" -u scripts/quick_eval_mcts.py \
        --agent "$1" --sims "$SIMS" --games "$GNUGO_GAMES" \
        --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
        --opponent gnugo:level=1 --pass-hack 2>&1 | tee -a "$LOG")
    GNUGO_PH_WINS=$(echo "$out" | sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p')
}

run_probe () {
    local out
    out=$("$PY" -u scripts/probe_policy_value.py --agent "$1" 2>&1 | tee -a "$LOG")
    POLICY_CE=$(echo "$out" | grep '^PROBE' | sed 's/.*policy_ce=\([0-9.]*\).*/\1/')
    VALUE_ACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')
}

# ── Round 0: seed baseline ───────────────────────────────────────────
if ! grep -q '^0,' "$RESULTS_CSV"; then
    log "round 0: seeding from $SEED_AGENT"
    cp "$SEED_AGENT" "$ROUND0_AGENT"
    run_gnugo "$ROUND0_AGENT"
    run_gnugo_passhack "$ROUND0_AGENT"
    run_probe "$ROUND0_AGENT"
    echo "0,10,${H2H_GAMES},${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},,,,,${GNUGO_PH_WINS}" >> "$RESULTS_CSV"
    log "round 0: baseline gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ph=${GNUGO_PH_WINS}/${GNUGO_GAMES} ce=${POLICY_CE} vacc=${VALUE_ACC}"
fi

# ── Main loop ────────────────────────────────────────────────────────
for round in $(seq 1 "$ROUNDS"); do
    R=$(printf '%03d' "$round")
    PREV=$(printf '%03d' "$((round - 1))")
    LATEST_AGENT="$CKPT_DIR/round_${PREV}_agent.h5"
    ROUND_MODEL="$CKPT_DIR/round_${R}.h5"
    ROUND_AGENT="$CKPT_DIR/round_${R}_agent.h5"
    ROUND_DATA="$DATA_DIR/round_${R}"

    if grep -q "^${round}," "$RESULTS_CSV"; then
        log "round $round: SKIP (already in CSV)"
        continue
    fi

    TRAIN_LR=$(get_lr "$round")

    # ── Stage 1: self-play generation ────────────────────────────────
    T_GEN0=$(date +%s)
    if ls "$ROUND_DATA"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
        log "round $round stage1: SKIP (chunks exist)"
    else
        mkdir -p "$ROUND_DATA"
        log "round $round stage1: $GAMES_PER_ROUND games (gumbel n=$SIMS m=$GUMBEL_M, agent=round_${PREV})"
        "$PY" -u scripts/generate_selfplay_data_mp.py \
            --agent "$LATEST_AGENT" \
            --num-games "$GAMES_PER_ROUND" \
            --num-workers "$WORKERS" \
            --sims "$SIMS" \
            --search gumbel --gumbel-m "$GUMBEL_M" \
            --value-target outcome \
            --board-size "$BOARD" \
            --batch-mcts "$BATCH_MCTS" \
            --chunk-size 250 \
            --seed "$((1000 + round))" \
            --onnx \
            --out-dir "$ROUND_DATA" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage1: FAILED rc=$rc"; exit 1; fi
    fi
    T_GEN=$(( $(date +%s) - T_GEN0 ))

    # ── Stage 2: training (growing window, capped at MAX_WINDOW) ─────
    if [ -f "$ROUND_AGENT" ]; then
        log "round $round stage2: SKIP (checkpoint exists)"
    else
        TRAIN_GLOBS=()
        WINDOW_SIZE=$MAX_WINDOW
        [ "$WINDOW_SIZE" -gt "$round" ] && WINDOW_SIZE=$round
        FIRST=$(( round - WINDOW_SIZE + 1 ))
        for r in $(seq "$FIRST" "$round"); do
            TRAIN_GLOBS+=("$DATA_DIR/round_$(printf '%03d' "$r")/selfplay_b${BOARD}_*.h5")
        done
        # Warmup: pad with bootstrap chunks for early rounds only.
        if [ "$round" -eq 1 ]; then
            PAD=$(ls "$BOOTSTRAP_GLOB_DIR"/selfplay_b9_*.h5 | head -2)
        elif [ "$round" -le 3 ]; then
            PAD=$(ls "$BOOTSTRAP_GLOB_DIR"/selfplay_b9_*.h5 | sed -n "$((round + 1))p")
        else
            PAD=""
        fi
        for f in $PAD; do TRAIN_GLOBS+=("$f"); done

        log "round $round stage2: train (init=round_${PREV}, window=${WINDOW_SIZE}, ${#TRAIN_GLOBS[@]} globs, lr=$TRAIN_LR)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from "$LATEST_AGENT" --full-init \
            --train-glob "${TRAIN_GLOBS[@]}" \
            --dst "$ROUND_MODEL" \
            --num-blocks "$BLOCKS" --filters "$FILTERS" \
            --board-size "$BOARD" --heads 2 --augment \
            --epochs "$TRAIN_EPOCHS" --bs "$TRAIN_BS" --lr "$TRAIN_LR" \
            --patience 99 --mega-chunks 4 \
            --policy-weight 1.0 --value-weight 1.0 \
            --seed "$((1 + round))" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage2: FAILED rc=$rc"; exit 1; fi
        if [ ! -f "$ROUND_AGENT" ]; then
            log "round $round stage2: FAILED (no agent checkpoint produced)"
            exit 1
        fi
    fi

    # ── Stage 3: evaluation ──────────────────────────────────────────
    POSITIONS=$("$PY" - <<EOF
import glob, h5py
n = 0
for p in glob.glob("$ROUND_DATA/selfplay_b${BOARD}_*.h5"):
    with h5py.File(p) as f:
        n += int(f.attrs.get('num_positions', 0))
print(n)
EOF
)

    H2H0_WINS=""
    GNUGO_WINS=""
    GNUGO_PH_WINS=""
    POLICY_CE=""
    VALUE_ACC=""
    H2HPREV_WINS=""
    H2HPREV_N=""

    if [ $((round % EVAL_EVERY)) -eq 0 ]; then
        log "round $round stage3: gnugo eval ($GNUGO_GAMES games)"
        run_gnugo "$ROUND_AGENT"
        log "round $round stage3: gnugo ${GNUGO_WINS}/${GNUGO_GAMES}"

        log "round $round stage3: gnugo+passhack eval ($GNUGO_GAMES games)"
        run_gnugo_passhack "$ROUND_AGENT"
        log "round $round stage3: gnugo+ph ${GNUGO_PH_WINS}/${GNUGO_GAMES}"

        run_probe "$ROUND_AGENT"

        if [ $((round % 10)) -eq 0 ] && [ "$round" -ge 10 ]; then
            REF=$(printf '%03d' "$((round - 10))")
            log "round $round stage3: rolling h2h vs round_${REF}"
            run_h2h "$ROUND_AGENT" "$CKPT_DIR/round_${REF}_agent.h5" "$((200 + round))"
            H2HPREV_WINS=$H2H_WINS
            H2HPREV_N=$H2H_GAMES
        fi

        log "round $round: COMPLETE (gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ph=${GNUGO_PH_WINS}/${GNUGO_GAMES} ce=${POLICY_CE} vacc=${VALUE_ACC} lr=${TRAIN_LR})"
    else
        log "round $round: COMPLETE (train-only, lr=${TRAIN_LR})"
    fi

    echo "${round},${H2H0_WINS},${H2H_GAMES},${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},${H2HPREV_WINS},${H2HPREV_N},${POSITIONS},${T_GEN},${GNUGO_PH_WINS}" >> "$RESULTS_CSV"
done

log "ALL ROUNDS DONE"
"$PY" -u scripts/plot_gumbel_loop.py --csv "$RESULTS_CSV" \
    --out results/gumbel_loop_b9_v3.png >> "$LOG" 2>&1 || true
cat "$RESULTS_CSV"
