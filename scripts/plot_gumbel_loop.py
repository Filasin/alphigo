#!/usr/bin/env python3
"""Plot the Gumbel self-play loop curves from results CSV."""
import argparse
import csv
import math


def wilson(wins, n, z=1.96):
    """Wilson score interval -> (center, half_width) in percent."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center * 100, half * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rounds, h2h_c, h2h_e, gg_c, gg_e, ce, vacc = [], [], [], [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row['round']))
            c, e = wilson(int(row['h2h0_wins']), int(row['h2h0_n']))
            h2h_c.append(c); h2h_e.append(e)
            c, e = wilson(int(row['gnugo_wins']), int(row['gnugo_n']))
            gg_c.append(c); gg_e.append(e)
            ce.append(float(row['policy_ce']))
            vacc.append(float(row['value_acc']) * 100)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.errorbar(rounds, h2h_c, yerr=h2h_e, marker='o', capsize=3,
                 label='h2h vs round 0 (primary)')
    ax1.errorbar(rounds, gg_c, yerr=gg_e, marker='s', capsize=3,
                 label='vs GnuGo L1 (ceiling marker)')
    ax1.axhline(50, color='gray', ls='--', lw=0.8)
    ax1.set_ylabel('win % (Wilson 95% CI)')
    ax1.set_ylim(-5, 105)
    ax1.legend()
    ax1.set_title('Gumbel self-play loop, 9x9 (n=32, m=16)')

    ax2.plot(rounds, ce, marker='o', label='policy CE on OGS probe')
    ax2.set_ylabel('policy cross-entropy', color='tab:blue')
    ax3 = ax2.twinx()
    ax3.plot(rounds, vacc, marker='s', color='tab:orange',
             label='value sign-acc %')
    ax3.set_ylabel('value sign-accuracy %', color='tab:orange')
    ax2.set_xlabel('round')

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
