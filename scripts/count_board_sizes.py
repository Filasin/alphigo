import sys, json, gzip
from collections import Counter

path = sys.argv[1]
sizes = Counter()
with gzip.open(path, 'rt') as f:
    for line in f:
        d = json.loads(line)
        w, h = d.get('width'), d.get('height')
        sizes[(w, h)] += 1

for k, v in sizes.most_common(15):
    print(f"{k[0]}x{k[1]}: {v:,}")
print(f"\nTotal: {sum(sizes.values()):,}")
