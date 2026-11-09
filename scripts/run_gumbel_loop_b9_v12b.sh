#!/bin/bash
# Gumbel AlphaZero self-play loop on 9x9 — v12b (search-quality intervention).
# Continues from the FROZEN v12 checkpoint with a doubled search budget:
# 512 sims, m=32. Hypothesis (from the v12 audit): the ceiling is search-target
# quality, not capacity — train CE == fresh-selfplay CE (zero generalization
# gap) at r51, so better search targets should move the ceiling where more
# rounds at 256/16 no longer do.
#
# Changes vs v12:
#   - Init: frozen v12 weights (INIT_AGENT), not a fresh random model
#   - SIMS 256->512, GUMBEL_M 16->32 (better policy targets per position)
#   - Throughput: WORKERS 10->13, GPU_BATCH_WAIT_MS 0->2 (GPU was ~33% util)
#   - Pachi eval @16000 playouts (8000 was near-saturated: 17/20 at r55)
#   - Yardstick = the frozen v12 init itself (v11-r132 saturated at 90%)
#   - edge-v3 puzzle suite (n=20, D4 rotations) added to the mod-20 cadence
#
# Usage:
#   bash scripts/run_gumbel_loop_b9_v12b.sh [--rounds N] [--games N] [--init-agent PATH]
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
TRT_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/tensorrt_libs"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${TRT_LIB}:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config ───────────────────────────────────────────────────────────
BOARD=9
BLOCKS=8
FILTERS=64
VALUE_FC=64
SIMS=512
GUMBEL_M=32
GAMES_PER_ROUND=3000
WORKERS=13
BATCH_MCTS=128
TRAIN_EPOCHS=1
TRAIN_BS=512
ROUNDS=10
MAX_WINDOW=10
MIN_WINDOW=5
EVAL_EVERY=1
H2H_GAMES=20
GNUGO_GAMES=20
# Frozen v12 checkpoint this run continues from. Set at freeze time (r60
# decision); override with --init-agent. Also used as the fixed yardstick:
# h2h vs the starting weights = net gain of the 512/32 intervention.
INIT_AGENT="${V12B_INIT_AGENT:-runs/gumbel_loop_b9_v12/checkpoints/round_080_agent.h5}"
YARDSTICK_GAMES=40
# Pachi @16000: @8000 was near-saturated by v12 r55 (17/20). 4x playout
# step ~ 450-500 Elo on 9x9, so 16000 restores two-sided resolution.
PACHI_OPP="pachi:playouts=16000"
PACHI_GAMES=20
BN_MOMENTUM=0.9
USE_ONNX=true
WEIGHT_DECAY=1e-5
VALUE_WEIGHT=1.5
OWNERSHIP_WEIGHT=0.5
SCORE_WEIGHT=0.5
PLAYOUT_CAP_FRAC=0.75
HEADS=4
Q_BLEND=0.0
SCORE_MARGIN_SCALE=10
ENCODER=alphazero_komi
KOMI_LO=5.5
KOMI_HI=9.5

RUN_DIR=runs/gumbel_loop_b9_v12b
DATA_DIR=${ALPHIGO_DATA_DIR:-data}/gumbel_loop_b9_v12b
RESULTS_CSV=results/gumbel_loop_b9_v12b.csv
LOG=logs/gumbel_loop_b9_v12b.log

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --games) GAMES_PER_ROUND="$2"; shift 2 ;;
        --init-agent) INIT_AGENT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[gumbel-v12b] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

CKPT_DIR="$RUN_DIR/checkpoints"
ROUND0_AGENT="$CKPT_DIR/round_000_agent.h5"
YARDSTICK_AGENT="$ROUND0_AGENT"

if [ ! -f "$INIT_AGENT" ] && [ ! -f "$ROUND0_AGENT" ]; then
    echo "FATAL: init agent not found: $INIT_AGENT"
    echo "Set --init-agent (or V12B_INIT_AGENT) to the frozen v12 checkpoint."
    exit 1
fi

# ── CSV header ───────────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,yard_wins,yard_n,gnugo_wins,gnugo_n,policy_ce,value_acc,self_vacc,h2hprev_wins,h2hprev_n,positions,gen_secs,gnugo_ph_wins,pachi_wins" > "$RESULTS_CSV"
fi

# ── LR schedule (step decay of the 1cycle base LR) ──────────────────
# v12 finding: h2h deceleration was LR-independent, and 0.01 was held from
# r52 via lr_hold.txt. v12b starts at 0.01; lr_hold.txt (in this run's
# checkpoints dir) overrides live, same mechanics as v12.
get_lr() {
    local r=$1
    local lr_hold="$CKPT_DIR/lr_hold.txt"
    if [ -f "$lr_hold" ]; then
        local hold_round hold_lr
        read -r hold_round hold_lr < "$lr_hold"
        if [ "$r" -ge "$hold_round" ]; then
            echo "$hold_lr"
            return
        fi
    fi
    if [ "$r" -lt 40 ]; then
        echo "0.010000"
    elif [ "$r" -lt 80 ]; then
        echo "0.003000"
    elif [ "$r" -lt 120 ]; then
        echo "0.001000"
    else
        echo "0.000300"
    fi
}

# ── Helpers ──────────────────────────────────────────────────────────
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

run_puzzles () {
    log "round $2 stage3: tactical puzzle test v1 (sims=$SIMS)"
    "$PY" -u scripts/_test_tactics.py --agent "$1" --sims "$SIMS" 2>&1 | tee -a "$LOG"
    log "round $2 stage3: edge puzzle test v2 (sims=$SIMS)"
    "$PY" -u scripts/_test_tactics_edge_v2.py --agent "$1" --sims "$SIMS" 2>&1 | tee -a "$LOG"
    # edge-v3: 5 engine-verified archetypes x 4 D4 rotations (n=20); the
    # primary instrument for the score-dist-head / edge-tactics question.
    log "round $2 stage3: edge puzzle test v3 (sims=$SIMS m=$GUMBEL_M)"
    "$PY" -u scripts/_test_tactics_edge_v3.py --agent "$1" --sims "$SIMS" \
        --gumbel-m "$GUMBEL_M" 2>&1 | tee -a "$LOG"
}

# ── Round 0: seed from the frozen v12 checkpoint ─────────────────────
if ! grep -q '^0,' "$RESULTS_CSV"; then
    if [ ! -f "$ROUND0_AGENT" ]; then
        log "round 0: seeding from frozen v12 checkpoint: $INIT_AGENT"
        cp "$INIT_AGENT" "$ROUND0_AGENT"
    fi
    run_gnugo "$ROUND0_AGENT"
    run_gnugo_passhack "$ROUND0_AGENT"
    run_probe "$ROUND0_AGENT"
    echo "0,,,${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},,,,,,${GNUGO_PH_WINS}," >> "$RESULTS_CSV"
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

    # ── Stage 1: self-play generation (randomized komi) ──────────────
    T_GEN0=$(date +%s)
    if ls "$ROUND_DATA"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
        log "round $round stage1: SKIP (chunks exist)"
    else
        mkdir -p "$ROUND_DATA"
        log "round $round stage1: $GAMES_PER_ROUND games (gumbel n=$SIMS m=$GUMBEL_M pcr=$PLAYOUT_CAP_FRAC komi=${KOMI_LO}-${KOMI_HI}, agent=round_${PREV})"
        ONNX_FLAG=""
        [ "$USE_ONNX" = "true" ] && ONNX_FLAG="--onnx"
        ORT_INTRA_OP_NUM_THREADS=2 \
        ORT_INTER_OP_NUM_THREADS=1 \
        TF_NUM_INTRAOP_THREADS=1 \
        PLAYOUT_CAP_FRAC="$PLAYOUT_CAP_FRAC" \
        SIMS_LOW=32 \
        GPU_BATCH_WAIT_MS=2 \
        WORKER_GAMES=1 \
        "$PY" -u scripts/generate_selfplay_data_mp.py \
            --agent "$LATEST_AGENT" \
            --num-games "$GAMES_PER_ROUND" \
            --num-workers "$WORKERS" \
            --sims "$SIMS" \
            --search gumbel --gumbel-m "$GUMBEL_M" \
            --value-target margin --score-margin-scale "$SCORE_MARGIN_SCALE" \
            --komi-random "$KOMI_LO" "$KOMI_HI" \
            --board-size "$BOARD" \
            --batch-mcts "$BATCH_MCTS" \
            --chunk-size 250 \
            --seed "$((20000 + round))" \
            $ONNX_FLAG \
            --out-dir "$ROUND_DATA" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage1: FAILED rc=$rc"; exit 1; fi
    fi
    T_GEN=$(( $(date +%s) - T_GEN0 ))

    # ── Stage 2: training (4-head, margin+scoredist targets, 1cycle) ─
    if [ -f "$ROUND_AGENT" ]; then
        log "round $round stage2: SKIP (checkpoint exists)"
    else
        TRAIN_GLOBS=()
        WINDOW_SIZE=$(( MIN_WINDOW + round / 2 ))
        [ "$WINDOW_SIZE" -gt "$MAX_WINDOW" ] && WINDOW_SIZE=$MAX_WINDOW
        [ "$WINDOW_SIZE" -gt "$round" ] && WINDOW_SIZE=$round
        FIRST=$(( round - WINDOW_SIZE + 1 ))
        for r in $(seq "$FIRST" "$round"); do
            TRAIN_GLOBS+=("$DATA_DIR/round_$(printf '%03d' "$r")/selfplay_b${BOARD}_*.h5")
        done

        log "round $round stage2: train (init=round_${PREV}, window=${WINDOW_SIZE}, lr=$TRAIN_LR 1cycle, wd=$WEIGHT_DECAY, bs=$TRAIN_BS, vw=$VALUE_WEIGHT, ow=$OWNERSHIP_WEIGHT, sw=$SCORE_WEIGHT, heads=$HEADS, enc=$ENCODER)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from "$LATEST_AGENT" --full-init \
            --train-glob "${TRAIN_GLOBS[@]}" \
            --dst "$ROUND_MODEL" \
            --num-blocks "$BLOCKS" --filters "$FILTERS" \
            --value-fc "$VALUE_FC" \
            --board-size "$BOARD" --heads "$HEADS" --augment \
            --encoder "$ENCODER" \
            --epochs "$TRAIN_EPOCHS" --bs "$TRAIN_BS" --lr "$TRAIN_LR" \
            --weight-decay "$WEIGHT_DECAY" \
            --bn-momentum "$BN_MOMENTUM" \
            --patience 99 --mega-chunks 8 \
            --policy-weight 1.0 --value-weight "$VALUE_WEIGHT" \
            --ownership-weight "$OWNERSHIP_WEIGHT" \
            --score-weight "$SCORE_WEIGHT" \
            --q-blend "$Q_BLEND" \
            --one-cycle \
            --seed "$((1 + round))" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage2: FAILED rc=$rc"; exit 1; fi
        if [ ! -f "$ROUND_AGENT" ]; then
            log "round $round stage2: FAILED (no agent checkpoint produced)"
            exit 1
        fi
    fi

    # ── Stage 3: evaluation (sequential) ────────────────────────────────
    POSITIONS=$("$PY" - <<EOF
import glob, h5py
n = 0
for p in glob.glob("$ROUND_DATA/selfplay_b${BOARD}_*.h5"):
    with h5py.File(p) as f:
        n += int(f.attrs.get('num_positions', 0))
print(n)
EOF
)

    if [ $((round % EVAL_EVERY)) -eq 0 ]; then
        _GNUGO_WINS=""
        _GNUGO_PH_WINS=""
        _POLICY_CE=""
        _VALUE_ACC=""
        _SELF_VACC=""
        _H2HPREV_WINS=""
        _H2HPREV_N=""
        _YARD_WINS=""
        _YARD_N=""
        _PACHI_WINS=""

        _H2H_PID=""
        if [ $((round % 10)) -eq 0 ] && [ "$round" -ge 10 ]; then
            REF=$(printf '%03d' "$((round - 10))")
            log "round $round stage3: rolling h2h vs round_${REF} (parallel)"
            "$PY" -u scripts/head_to_head.py \
                --agent-a "$ROUND_AGENT" --agent-b "$CKPT_DIR/round_${REF}_agent.h5" \
                --num-games "$H2H_GAMES" --num-workers 4 --board-size "$BOARD" \
                --sims "$SIMS" --search gumbel --gumbel-m "$GUMBEL_M" \
                --seed "$((200 + round))" > "/tmp/v12b_h2h_r${round}.log" 2>&1 &
            _H2H_PID=$!
        fi

        if [ $((round % 5)) -eq 0 ]; then
            log "round $round stage3: gnugo evals (2x$GNUGO_GAMES games) + pachi ($PACHI_OPP, $PACHI_GAMES games), parallel"
            "$PY" -u scripts/quick_eval_mcts.py \
                --agent "$ROUND_AGENT" --sims "$SIMS" --games "$PACHI_GAMES" \
                --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
                --opponent "$PACHI_OPP" > "/tmp/v12b_pachi_r${round}.log" 2>&1 &
            _PID_PACHI=$!
            "$PY" -u scripts/quick_eval_mcts.py \
                --agent "$ROUND_AGENT" --sims "$SIMS" --games "$GNUGO_GAMES" \
                --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
                --opponent gnugo:level=1 > "/tmp/v12b_gnugo_plain_r${round}.log" 2>&1 &
            _PID_PLAIN=$!
            "$PY" -u scripts/quick_eval_mcts.py \
                --agent "$ROUND_AGENT" --sims "$SIMS" --games "$GNUGO_GAMES" \
                --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
                --opponent gnugo:level=1 --pass-hack > "/tmp/v12b_gnugo_ph_r${round}.log" 2>&1 &
            _PID_PH=$!
            wait "$_PID_PLAIN" "$_PID_PH" "$_PID_PACHI"
            cat "/tmp/v12b_gnugo_plain_r${round}.log" "/tmp/v12b_gnugo_ph_r${round}.log" "/tmp/v12b_pachi_r${round}.log" >> "$LOG"
            _GNUGO_WINS=$(sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p' "/tmp/v12b_gnugo_plain_r${round}.log")
            _GNUGO_PH_WINS=$(sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p' "/tmp/v12b_gnugo_ph_r${round}.log")
            _PACHI_WINS=$(sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p' "/tmp/v12b_pachi_r${round}.log")
            log "round $round stage3: gnugo ${_GNUGO_WINS}/${GNUGO_GAMES} ph ${_GNUGO_PH_WINS}/${GNUGO_GAMES} pachi ${_PACHI_WINS}/${PACHI_GAMES}"
            rm -f "/tmp/v12b_gnugo_plain_r${round}.log" "/tmp/v12b_gnugo_ph_r${round}.log" "/tmp/v12b_pachi_r${round}.log"
        fi

        out=$("$PY" -u scripts/probe_policy_value.py --agent "$ROUND_AGENT" 2>&1 | tee -a "$LOG")
        _POLICY_CE=$(echo "$out" | grep '^PROBE' | sed 's/.*policy_ce=\([0-9.]*\).*/\1/')
        _VALUE_ACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')

        SELF_CHUNK=$(ls "$ROUND_DATA"/selfplay_b${BOARD}_*.h5 2>/dev/null | tail -1)
        if [ -n "$SELF_CHUNK" ]; then
            out=$("$PY" -u scripts/probe_policy_value.py --agent "$ROUND_AGENT" \
                --probe-chunk "$SELF_CHUNK" 2>&1 | tee -a "$LOG")
            _SELF_VACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')
        fi

        if [ -n "$_H2H_PID" ]; then
            wait "$_H2H_PID"
            cat "/tmp/v12b_h2h_r${round}.log" >> "$LOG"
            _H2HPREV_WINS=$(sed -n 's/.*A wins: \([0-9][0-9]*\).*/\1/p' "/tmp/v12b_h2h_r${round}.log")
            _H2HPREV_N=$H2H_GAMES
            rm -f "/tmp/v12b_h2h_r${round}.log"
        fi

        if [ $((round % 20)) -eq 0 ]; then
            if [ -f "$YARDSTICK_AGENT" ]; then
                log "round $round stage3: yardstick h2h vs v12b start weights ($YARDSTICK_GAMES games)"
                "$PY" -u scripts/head_to_head.py \
                    --agent-a "$ROUND_AGENT" --agent-b "$YARDSTICK_AGENT" \
                    --num-games "$YARDSTICK_GAMES" --num-workers 4 --board-size "$BOARD" \
                    --sims "$SIMS" --search gumbel --gumbel-m "$GUMBEL_M" \
                    --seed "$((900 + round))" > "/tmp/v12b_yard_r${round}.log" 2>&1
                cat "/tmp/v12b_yard_r${round}.log" >> "$LOG"
                _YARD_WINS=$(sed -n 's/.*A wins: \([0-9][0-9]*\).*/\1/p' "/tmp/v12b_yard_r${round}.log")
                _YARD_N=$YARDSTICK_GAMES
                log "round $round stage3: yardstick vs start = ${_YARD_WINS}/${YARDSTICK_GAMES}"
                rm -f "/tmp/v12b_yard_r${round}.log"
            fi
            run_puzzles "$ROUND_AGENT" "$round"
        fi

        log "round $round: COMPLETE (gnugo=${_GNUGO_WINS}/${GNUGO_GAMES} ph=${_GNUGO_PH_WINS}/${GNUGO_GAMES} pachi=${_PACHI_WINS}/${PACHI_GAMES} ce=${_POLICY_CE} vacc=${_VALUE_ACC} svacc=${_SELF_VACC} lr=${TRAIN_LR} wd=${WEIGHT_DECAY})"
        echo "${round},${_YARD_WINS},${_YARD_N},${_GNUGO_WINS},${GNUGO_GAMES},${_POLICY_CE},${_VALUE_ACC},${_SELF_VACC},${_H2HPREV_WINS},${_H2HPREV_N},${POSITIONS},${T_GEN},${_GNUGO_PH_WINS},${_PACHI_WINS}" >> "$RESULTS_CSV"
    else
        log "round $round: COMPLETE (train-only, lr=${TRAIN_LR})"
        echo "${round},,,,,,,,,,${POSITIONS},${T_GEN},," >> "$RESULTS_CSV"
    fi
done

log "ALL ROUNDS DONE"
"$PY" -u scripts/plot_gumbel_loop.py --csv "$RESULTS_CSV" \
    --out results/gumbel_loop_b9_v12b.png >> "$LOG" 2>&1 || true
cat "$RESULTS_CSV"
