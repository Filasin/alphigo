import sys, json, gzip
from collections import Counter

path = sys.argv[1]
ranks = Counter()
ranked_count = 0
total_9x9 = 0
has_moves = 0

with gzip.open(path, 'rt') as f:
    for line in f:
        d = json.loads(line)
        if d.get('width') != 9 or d.get('height') != 9:
            continue
        total_9x9 += 1

        moves = d.get('moves', [])
        if len(moves) >= 20:
            has_moves += 1

        players = d.get('players', {})
        min_rank = None
        for color in ['black', 'white']:
            p = players.get(color, {})
            rank = p.get('rank')
            if rank is not None:
                r = int(rank)
                if min_rank is None or r < min_rank:
                    min_rank = r
        if min_rank is not None:
            ranked_count += 1
            ranks[min_rank] += 1

print(f"Total 9x9: {total_9x9:,}")
print(f"With rank info: {ranked_count:,}")
print(f"With 20+ moves: {has_moves:,}")
print(f"\nRank distribution (min of both players):")
print(f"{'Rank':>6} {'Approx level':>15} {'Games':>10}")
print("-" * 35)
for r in sorted(ranks.keys()):
    if r < 30:
        level = f"{30-r}k"
    else:
        level = f"{r-29}d"
    if ranks[r] >= 50:
        print(f"{r:>6} {level:>15} {ranks[r]:>10,}")

print(f"\n--- Strength buckets (min rank of both players) ---")
beginner = sum(v for k,v in ranks.items() if k < 10)
weak = sum(v for k,v in ranks.items() if 10 <= k < 20)
mid = sum(v for k,v in ranks.items() if 20 <= k < 25)
strong = sum(v for k,v in ranks.items() if 25 <= k < 30)
dan = sum(v for k,v in ranks.items() if k >= 30)
print(f"Beginner (<20k, rank<10): {beginner:,}")
print(f"Weak (20k-11k, rank 10-19): {weak:,}")
print(f"Mid (10k-6k, rank 20-24): {mid:,}")
print(f"Strong (5k-1k, rank 25-29): {strong:,}")
print(f"Dan+ (rank>=30): {dan:,}")

total_strong = strong + dan
pct = 100 * total_strong / total_9x9 if total_9x9 else 0
print(f"\nStrong+Dan total: {total_strong:,} ({pct:.1f}% of all 9x9)")
print(f"Extrapolated from 27M games: ~{int(total_strong / total_9x9 * 27_000_000 * 0.39):,} strong 9x9 games")
