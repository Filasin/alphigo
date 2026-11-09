"""Analyze game patterns from the selfplay experiment log."""
import re
import numpy as np

lines = open("logs/selfplay_value_experiment.log").readlines()

games = []
for line in lines:
    m = re.search(
        r"\[w(\d+) g(\d+)\] winner=(\w) reason=(\w+) moves=(\d+) "
        r"AZ (won|lost)\s+\((\d+)s\)", line)
    if m:
        games.append({
            "worker": int(m.group(1)),
            "gid": int(m.group(2)),
            "winner": m.group(3),
            "reason": m.group(4),
            "moves": int(m.group(5)),
            "az_won": m.group(6) == "won",
            "seconds": int(m.group(7)),
        })

# Assign sims levels by cumulative count (30+30+30+60+60)
sims_map = []
idx = 0
for sims, n in [(1, 30), (25, 30), (50, 30), (200, 60), (800, 60)]:
    for _ in range(min(n, len(games) - idx)):
        sims_map.append(sims)
        idx += 1

for sims in [1, 25, 50, 200, 800]:
    sg = [g for g, s in zip(games, sims_map) if s == sims]
    if not sg:
        continue
    wins = [g for g in sg if g["az_won"]]
    losses = [g for g in sg if not g["az_won"]]

    print(f"\n{'=' * 60}")
    print(f"SIMS={sims}: {len(sg)} games, {len(wins)} wins "
          f"({100 * len(wins) / len(sg):.1f}%)")
    print(f"{'=' * 60}")

    if wins:
        w_moves = [g["moves"] for g in wins]
        w_reasons = set(g["reason"] for g in wins)
        w_secs = [g["seconds"] for g in wins]
        print(f"\n  WINS ({len(wins)}):")
        print(f"    Reason:  {w_reasons}")
        print(f"    Moves:   min={min(w_moves)} max={max(w_moves)} "
              f"mean={np.mean(w_moves):.0f} median={np.median(w_moves):.0f}")
        print(f"    Time:    {min(w_secs)}-{max(w_secs)}s")

    if losses:
        l_moves = [g["moves"] for g in losses]
        l_reasons = set(g["reason"] for g in losses)
        l_secs = [g["seconds"] for g in losses]
        print(f"\n  LOSSES ({len(losses)}):")
        print(f"    Reason:  {l_reasons}")
        print(f"    Moves:   min={min(l_moves)} max={max(l_moves)} "
              f"mean={np.mean(l_moves):.0f} median={np.median(l_moves):.0f}")
        print(f"    Time:    {min(l_secs)}-{max(l_secs)}s")

    if wins and losses:
        gap = min(l_moves) - max(w_moves)
        if gap > 0:
            print(f"\n  BIMODAL GAP: {gap} moves between longest win "
                  f"({max(w_moves)}) and shortest loss ({min(l_moves)})")
        else:
            print(f"\n  OVERLAP: win/loss move ranges overlap")
