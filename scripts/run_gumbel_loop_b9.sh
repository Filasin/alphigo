#!/bin/bash
# Gumbel AlphaZero self-play loop on 9x9 (no gating, continuous training).
#
#   round 0: seed = C++ bootstrap model, eval only
#   round r: selfplay (gumbel n=32, m=16) -> train (window 4 rounds,
#            warmup-padded with bootstrap chunks) -> eval (h2h vs round 0
#            primary, GnuGo L1 secondary, probe, every 5: h2h vs r-5)
#
# Usage:
#   bash scripts/run_gumbel_loop_b9.sh [--rounds N] [--games N] [--fresh]
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config ───────────────────────────────────────────────────────────
BOARD=9
BLOCKS=5
FILTERS=96
SIMS=32
GUMBEL_M=16
GAMES_PER_ROUND=1000
WORKERS=8
BATCH_MCTS=16
TRAIN_EPOCHS=2
TRAIN_BS=1024
TRAIN_LR=1e-4
ROUNDS=20
WINDOW=4
H2H_GAMES=20
GNUGO_GAMES=20
SEED_AGENT=runs/bootstrap_b9_scratch/model_bootstrap_agent.h5
BOOTSTRAP_GLOB_DIR=data/bootstrap_b9_h5

RUN_DIR=runs/gumbel_loop_b9
DATA_DIR=data/gumbel_loop_b9
RESULTS_CSV=results/gumbel_loop_b9.csv
LOG=logs/gumbel_loop_b9.log

# ── Parse args ───────────────────────────────────────────────────────
FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --games) GAMES_PER_ROUND="$2"; shift 2 ;;
        --fresh) FRESH=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$FRESH" -eq 1 ]; then
    rm -rf "$RUN_DIR" "$DATA_DIR" "$RESULTS_CSV"
fi
mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[gumbel-loop] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

CKPT_DIR="$RUN_DIR/checkpoints"
ROUND0_AGENT="$CKPT_DIR/round_000_agent.h5"

# ── CSV header ───────────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,h2h0_wins,h2h0_n,gnugo_wins,gnugo_n,policy_ce,value_acc,h2hprev_wins,h2hprev_n,positions,gen_secs" > "$RESULTS_CSV"
fi

# ── Helpers ──────────────────────────────────────────────────────────
run_h2h () {  # $1=agent_a  $2=agent_b  -> sets H2H_WINS
    local out
    out=$("$PY" -u scripts/head_to_head.py \
        --agent-a "$1" --agent-b "$2" \
        --num-games "$H2H_GAMES" --board-size "$BOARD" \
        --sims "$SIMS" --search gumbel --gumbel-m "$GUMBEL_M" \
        --seed "$3" 2>&1 | tee -a "$LOG")
    H2H_WINS=$(echo "$out" | sed -n 's/.*A wins: \([0-9][0-9]*\).*/\1/p')
}

run_gnugo () {  # $1=agent  -> sets GNUGO_WINS
    local out
    out=$("$PY" -u scripts/quick_eval_mcts.py \
        --agent "$1" --sims "$SIMS" --games "$GNUGO_GAMES" \
        --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
        --opponent gnugo:level=1 2>&1 | tee -a "$LOG")
    GNUGO_WINS=$(echo "$out" | sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p')
}

run_probe () {  # $1=agent -> sets POLICY_CE VALUE_ACC
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
    run_probe "$ROUND0_AGENT"
    echo "0,10,${H2H_GAMES},${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},,,," >> "$RESULTS_CSV"
    log "round 0: baseline gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ce=${POLICY_CE} vacc=${VALUE_ACC}"
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
            --out-dir "$ROUND_DATA" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage1: FAILED rc=$rc"; exit 1; fi
    fi
    T_GEN=$(( $(date +%s) - T_GEN0 ))

    # ── Stage 2: training (window + warmup padding) ──────────────────
    if [ -f "$ROUND_AGENT" ]; then
        log "round $round stage2: SKIP (checkpoint exists)"
    else
        TRAIN_GLOBS=()
        FIRST=$(( round - WINDOW + 1 )); [ "$FIRST" -lt 1 ] && FIRST=1
        for r in $(seq "$FIRST" "$round"); do
            TRAIN_GLOBS+=("$DATA_DIR/round_$(printf '%03d' "$r")/selfplay_b${BOARD}_*.h5")
        done
        # Warmup: pad with bootstrap chunks (r=1: 2 chunks, r=2,3: 1).
        if [ "$round" -eq 1 ]; then
            PAD=$(ls "$BOOTSTRAP_GLOB_DIR"/selfplay_b9_*.h5 | head -2)
        elif [ "$round" -le 3 ]; then
            PAD=$(ls "$BOOTSTRAP_GLOB_DIR"/selfplay_b9_*.h5 | sed -n "$((round + 1))p")
        else
            PAD=""
        fi
        for f in $PAD; do TRAIN_GLOBS+=("$f"); done

        log "round $round stage2: train (init=round_${PREV}, ${#TRAIN_GLOBS[@]} globs, lr=$TRAIN_LR)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from "$LATEST_AGENT" --full-init \
            --train-glob "${TRAIN_GLOBS[@]}" \
            --dst "$ROUND_MODEL" \
            --num-blocks "$BLOCKS" --filters "$FILTERS" \
            --board-size "$BOARD" --augment \
            --epochs "$TRAIN_EPOCHS" --bs "$TRAIN_BS" --lr "$TRAIN_LR" \
            --patience 99 --mega-chunks 4 \
            --policy-weight 1.0 --value-weight 1.0 --ownership-weight 0.5 \
            --seed "$((1 + round))" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage2: FAILED rc=$rc"; exit 1; fi
        if [ ! -f "$ROUND_AGENT" ]; then
            log "round $round stage2: FAILED (no agent checkpoint produced)"
            exit 1
        fi
    fi

    # ── Stage 3: evaluation (measurement only) ───────────────────────
    POSITIONS=$("$PY" - <<EOF
import glob, h5py
n = 0
for p in glob.glob("$ROUND_DATA/selfplay_b${BOARD}_*.h5"):
    with h5py.File(p) as f:
        n += int(f.attrs.get('num_positions', 0))
print(n)
EOF
)
    log "round $round stage3: h2h vs round 0 ($H2H_GAMES games)"
    run_h2h "$ROUND_AGENT" "$ROUND0_AGENT" "$((100 + round))"
    H2H0_WINS=$H2H_WINS
    log "round $round stage3: h2h0 ${H2H0_WINS}/${H2H_GAMES}"

    log "round $round stage3: gnugo eval ($GNUGO_GAMES games)"
    run_gnugo "$ROUND_AGENT"
    log "round $round stage3: gnugo ${GNUGO_WINS}/${GNUGO_GAMES}"

    run_probe "$ROUND_AGENT"

    H2HPREV_WINS=""
    H2HPREV_N=""
    if [ $((round % 5)) -eq 0 ] && [ "$round" -ge 5 ]; then
        REF=$(printf '%03d' "$((round - 5))")
        log "round $round stage3: rolling h2h vs round_${REF}"
        run_h2h "$ROUND_AGENT" "$CKPT_DIR/round_${REF}_agent.h5" "$((200 + round))"
        H2HPREV_WINS=$H2H_WINS
        H2HPREV_N=$H2H_GAMES
    fi

    echo "${round},${H2H0_WINS},${H2H_GAMES},${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},${H2HPREV_WINS},${H2HPREV_N},${POSITIONS},${T_GEN}" >> "$RESULTS_CSV"
    log "round $round: COMPLETE (h2h0=${H2H0_WINS}/${H2H_GAMES} gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ce=${POLICY_CE} vacc=${VALUE_ACC})"
done

log "ALL ROUNDS DONE"
"$PY" -u scripts/plot_gumbel_loop.py --csv "$RESULTS_CSV" \
    --out results/gumbel_loop_b9.png >> "$LOG" 2>&1 || true
cat "$RESULTS_CSV"
