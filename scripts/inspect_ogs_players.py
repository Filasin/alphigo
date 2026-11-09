import sys, json, gzip

path = sys.argv[1]
with gzip.open(path, 'rt') as f:
    count = 0
    for line in f:
        d = json.loads(line)
        if d.get('width') != 9 or d.get('height') != 9:
            continue
        players = d.get('players', {})
        print(json.dumps(players, indent=2))
        count += 1
        if count >= 3:
            break
