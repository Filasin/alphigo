"""Baseline opponent Elo registry and the external-evaluation helper.

This module owns the single canonical `REFERENCE_ELOS` mapping used by
both `train_rl.py` (per-round external eval) and the post-hoc
evaluation script. Previously each script kept its own copy;
they have been merged here so the numbers cannot drift.

Values are on the EGF scale (1 dan ~= 2100, each rank ~= 100 points).

Sources:
    GnuGo L10: ~8 kyu on KGS (Wikipedia, Sensei's Library) -> ~1200 EGF
    GnuGo levels: "huge reduction per level" -- estimated from forum reports
    Pachi: KGS 3k at 5k playouts, ~100 Elo per doubling
        (github.com/pasky/pachi)
    Fuego: KGS ~2 kyu (Fuego TCIAIG paper, CGOS standings)

Per CONTEXT.md: Elo here is a *relative* anchor used to convert win
rates against a fixed opponent into an approximate absolute strength
estimate. It is not claimed as a calibrated rating.
"""

REFERENCE_ELOS = {
    'gnugo': 1200,
    'gnugo:level=1': 600,
    'gnugo:level=3': 900,
    'gnugo:level=5': 1050,
    'gnugo:level=10': 1200,
    'fuego:max_games=1000': 1600,
    'fuego:max_games=5000': 1800,
    'pachi:playouts=500': 1200,
    'pachi:playouts=1000': 1400,
    'pachi:playouts=5000': 1700,
}


def evaluate_external(agent_path, board_size, eval_opponents, num_games):
    """Evaluate one agent checkpoint against external GTP opponents.

    Thin wrapper around `eval_checkpoints.evaluate_checkpoints` that
    injects `REFERENCE_ELOS` so absolute Elo estimates are filled in,
    and reshapes the row list into the dict shape `train_rl.py` writes
    to `experiments.jsonl`:

        {opponent_spec: {'win_rate': float in [0, 1],
                         'elo_delta': int|None,
                         'abs_elo':  int|None}}
    """
    import eval_checkpoints

    rows, _ = eval_checkpoints.evaluate_checkpoints(
        [agent_path],
        eval_opponents,
        num_games=num_games,
        board_size=board_size,
        reference_elos=REFERENCE_ELOS,
        verbose=True,
    )
    out = {}
    for row in rows:
        out[row['opponent']] = {
            'win_rate': row['win_pct'] / 100.0,
            'elo_delta': row['elo_diff'],
            'abs_elo': row.get('abs_elo'),
        }
    return out
