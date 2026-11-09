#!/usr/bin/env bash
# 9x9 supervised-learning sweep across encoder × network combos.
#
# Usage:   bash scripts/sl_sweep_9x9.sh
#
# Run inside WSL with the alphigo-venv active:
#   source ~/alphigo-venv/bin/activate
#   cd <repository root>
#   nohup bash scripts/sl_sweep_9x9.sh > logs/sl_sweep_9x9_main.log 2>&1 &
#   disown
#
# On first launch, any pre-existing supervised_<TAG>_*_b<BOARD>_*  files
# are MOVED to checkpoints/archive_pre_sweep_<timestamp>/ so the sweep
# starts from a clean slate.  Subsequent launches (e.g. after
# interruption) skip combos whose new history file already exists, so
# you can resume without redoing finished work.
#
# Per-run log lands in logs/sl_sweep/<encoder>_<network>.log.  Failures
# are recorded and the sweep continues.

set -u   # NOT -e: keep going on a single combo's failure.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs/sl_sweep checkpoints

TAG="9x9"                # checkpoint filename prefix: supervised_<TAG>_<enc>_<net>_*
DATA_DIR="data_9x9"
NUM_GAMES=8000           # train games drawn (sampler reserves 10% for test)
EPOCHS=100
PATIENCE=10
OPT="adam"
BOARD=9
SEED=42

# ── One-time archival of pre-existing 9x9 SL checkpoints ─────────────
# We treat the presence of an "ARCHIVE_DONE" marker as proof that the
# sweep has already moved the legacy files aside, so a re-launch after
# interruption does NOT re-archive freshly-written sweep files.
ARCHIVE_MARKER="logs/sl_sweep/.archive_done"
if [ ! -f "$ARCHIVE_MARKER" ]; then
  ARCHIVE_DIR="checkpoints/archive_pre_sweep_$(date +%Y%m%d_%H%M%S)"
  legacy=( $(ls checkpoints/supervised_${TAG}_*_b${BOARD}_* 2>/dev/null) \
           $(ls checkpoints/mcts_*_b${BOARD}_* 2>/dev/null) )
  if [ ${#legacy[@]} -gt 0 ]; then
    mkdir -p "$ARCHIVE_DIR"
    echo "Archiving ${#legacy[@]} pre-existing 9x9 SL files → $ARCHIVE_DIR"
    mv checkpoints/supervised_${TAG}_*_b${BOARD}_* "$ARCHIVE_DIR/" 2>/dev/null || true
    mv checkpoints/mcts_*_b${BOARD}_*             "$ARCHIVE_DIR/" 2>/dev/null || true
  fi
  touch "$ARCHIVE_MARKER"
fi

# 13 combos: 4 encoders × 3 networks + alphazero canonical.
COMBOS=(
  "oneplane:small"
  "oneplane:medium"
  "oneplane:large"
  "sevenplane:small"
  "sevenplane:medium"
  "sevenplane:large"
  "elevenplane:small"
  "elevenplane:medium"
  "elevenplane:large"
  "agplane:small"
  "agplane:medium"
  "agplane:large"
  "alphazero:alphazero"
)

SUMMARY="logs/sl_sweep/SUMMARY.txt"
echo "9x9 SL sweep started at $(date -Iseconds)" > "$SUMMARY"
echo "tag=$TAG  data_dir=$DATA_DIR  num_games=$NUM_GAMES  epochs=$EPOCHS  patience=$PATIENCE" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for spec in "${COMBOS[@]}"; do
  enc="${spec%%:*}"
  net="${spec##*:}"
  pretty="${enc}_${net}"
  log="logs/sl_sweep/${pretty}.log"
  history="checkpoints/${TAG}_${enc}_${net}_${OPT}_b${BOARD}_history.json"

  echo "════════════════════════════════════════════════════════════════"
  echo "[$pretty] $(date +%H:%M:%S)"
  if [ -f "$history" ]; then
    echo "  → already trained ($history exists), skipping"
    echo "[$pretty] SKIP (history exists)" >> "$SUMMARY"
    continue
  fi

  echo "  → logging to $log"
  start_ts=$(date +%s)

  python train.py \
    --board-size "$BOARD" \
    --encoder "$enc" \
    --network "$net" \
    --data-dir "$DATA_DIR" \
    --num-games "$NUM_GAMES" \
    --optimizer "$OPT" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --tag "$TAG" \
    --seed "$SEED" \
    > "$log" 2>&1
  rc=$?

  elapsed=$(( $(date +%s) - start_ts ))
  hh=$(( elapsed / 3600 )); mm=$(( (elapsed % 3600) / 60 )); ss=$(( elapsed % 60 ))

  if [ "$rc" -ne 0 ]; then
    echo "  → FAILED (exit $rc) after ${hh}h${mm}m${ss}s — see $log"
    echo "[$pretty] FAIL rc=$rc time=${hh}h${mm}m${ss}s" >> "$SUMMARY"
    continue
  fi

  if [ -f "$history" ]; then
    best=$(python - <<PY
import json, math
h = json.load(open('$history'))
vl = h.get('val_loss', [])
va = h.get('val_accuracy', [])
if not vl:
    print('NO_HISTORY')
else:
    i = min(range(len(vl)), key=lambda k: vl[k])
    print(f"epoch={i+1} val_loss={vl[i]:.4f} val_acc={va[i]*100:.2f}%")
PY
)
  else
    best="(no history file)"
  fi
  echo "  → done in ${hh}h${mm}m${ss}s  best: $best"
  echo "[$pretty] OK time=${hh}h${mm}m${ss}s best: $best" >> "$SUMMARY"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Sweep done at $(date -Iseconds)"
echo "Summary at $SUMMARY:"
echo ""
cat "$SUMMARY"

# Final ranking — best val_loss across every supervised_<TAG>_*_b<BOARD>_history.json
python - <<PY
import json, glob, math
print("\n=== Ranked by val_loss (lower = better) ===")
rows = []
for h in glob.glob("checkpoints/supervised_${TAG}_*_b${BOARD}_history.json"):
    try:
        d = json.load(open(h))
    except Exception:
        continue
    vl = d.get("val_loss", [])
    va = d.get("val_accuracy", [])
    if not vl:
        continue
    i = min(range(len(vl)), key=lambda k: vl[k])
    name = h.split("supervised_${TAG}_", 1)[1].split("_${OPT}_")[0]
    rows.append((vl[i], va[i], i + 1, name))
rows.sort()
unif = math.log(${BOARD} * ${BOARD})
print(f"  uniform val_loss={unif:.3f}, uniform val_acc={100/(${BOARD}*${BOARD}):.2f}%")
print(f"  {'rank':<5}{'combo':<28}{'best_ep':>8}{'val_loss':>11}{'val_acc':>10}")
for r, (loss, acc, ep, name) in enumerate(rows, 1):
    flag = "  STUCK" if loss > unif * 0.99 else ""
    print(f"  {r:<5}{name:<28}{ep:>8}{loss:>11.4f}{acc*100:>9.2f}%{flag}")
PY
