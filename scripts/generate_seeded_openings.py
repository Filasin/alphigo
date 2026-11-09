"""Generate seeded openings JSON from the edge-v3 tactical archetypes.

Produces 20 openings (5 bases x 4 D4 rotations) that force the selfplay
generator into positions the agent struggles with organically.
"""
import json
import sys
import os

BS = 9

XFORMS = [
    ('a', lambda c, r: (c, r)),
    ('b', lambda c, r: (10 - r, c)),
    ('c', lambda c, r: (10 - c, 10 - r)),
    ('d', lambda c, r: (r, 10 - c)),
]

BASES = [
    dict(name='semeai-capture',
         seq=[('B', 'd1'), ('W', 'd2'), ('B', 'e1'), ('W', 'e2'),
              ('B', 'f2'), ('W', 'f1'), ('B', 'b2'), ('W', 'h5')]),
    dict(name='edge-connect',
         seq=[('B', 'f1'), ('W', 'e1'), ('B', 'g1'), ('W', 'f2'),
              ('B', 'h2'), ('W', 'g2'), ('B', 'h3'), ('W', 'd5')]),
    dict(name='cut-capture',
         seq=[('B', 'a2'), ('W', 'b2'), ('B', 'c2'), ('W', 'c1'),
              ('B', 'b3'), ('W', 'e5'), ('B', 'd2'), ('W', 'f5')]),
    dict(name='semeai-2stone',
         seq=[('B', 'd9'), ('W', 'd8'), ('B', 'e9'), ('W', 'e8'),
              ('B', 'f9'), ('W', 'f8'), ('B', 'g8'), ('W', 'g9'),
              ('B', 'h8'), ('W', 'h9')]),
    dict(name='counter-capture',
         seq=[('B', 'b1'), ('W', 'b2'), ('B', 'c1'), ('W', 'c2'),
              ('B', 'd2'), ('W', 'd1'), ('B', 'e2'), ('W', 'a3'),
              ('B', 'b4'), ('W', 'a4')]),
]


def _parse(ms):
    return ord(ms[0]) - ord('a') + 1, int(ms[1:])


def _fmt(c, r):
    return chr(ord('a') + c - 1) + str(r)


def main():
    openings = []
    for base in BASES:
        for xf_name, xf in XFORMS:
            moves = []
            for color, coord in base['seq']:
                c, r = _parse(coord)
                tc, tr = xf(c, r)
                moves.append([color, _fmt(tc, tr)])
            openings.append({
                'name': f'{base["name"]}-{xf_name}',
                'moves': moves,
            })

    out = os.path.join(os.path.dirname(__file__), 'seeded_openings_edge_v3.json')
    with open(out, 'w') as f:
        json.dump(openings, f, indent=2)
    print(f'Wrote {len(openings)} openings to {out}')


if __name__ == '__main__':
    main()
