"""Rank every 9x9 SL run by best val_loss. Reads history JSONs."""
import glob, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
patterns = [
    "checkpoints/9x9_*_b9_history.json",          # tag=9x9 (current sweep)
    "checkpoints/supervised_9x9_*_b9_history.json",  # legacy tag=supervised_9x9
]
rows = []
for pat in patterns:
    for h in sorted((ROOT / pat).parent.glob(Path(pat).name)):
        try:
            d = json.loads(h.read_text())
        except Exception as e:
            print(f"skip {h.name}: {e}", file=sys.stderr); continue
        vl = d.get("val_loss") or []
        va = d.get("val_accuracy") or []
        if not vl: continue
        i = min(range(len(vl)), key=lambda k: vl[k])
        name = h.stem.replace("_adam_b9_history", "")
        rows.append((vl[i], va[i], i + 1, len(vl), name))

rows.sort()
uloss = math.log(81)
print(f"uniform val_loss={uloss:.3f}, uniform val_acc={100/81:.2f}%\n")
print(f"{'rk':<4}{'combo':<32}{'best_ep':>9}{'total':>7}{'val_loss':>11}{'val_acc':>10}")
print("-" * 73)
for r, (loss, acc, ep, tot, name) in enumerate(rows, 1):
    flag = "  STUCK" if loss > uloss * 0.99 else ""
    print(f"{r:<4}{name:<32}{ep:>9}{tot:>7}{loss:>11.4f}{acc*100:>9.2f}%{flag}")
