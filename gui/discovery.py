"""Auto-discover trained agents from runs/ and external GTP binaries.

Used by the GUI compare mode to populate a numbered picker without
requiring the user to type out checkpoint paths.
"""

from pathlib import Path
from typing import List, Tuple


def _method_from_slug(slug: str) -> str:
    """Slug format: '<method>_<encoder>_<network>_lr<lr>_b<size>[_seed<n>]'."""
    return slug.split('_', 1)[0]


def discover_run_agents(runs_dir: str = 'runs') -> List[dict]:
    """Find every `runs/*/checkpoints/round_*.h5` and group by run dir.

    Returns a list of dicts with keys:
        slug, method, latest_round, latest_path, all_rounds (list[int]).
    """
    base = Path(runs_dir)
    if not base.exists():
        return []
    out = []
    for run_dir in sorted(base.iterdir()):
        ckpt_dir = run_dir / 'checkpoints'
        if not ckpt_dir.is_dir():
            continue
        rounds = []
        for f in ckpt_dir.glob('round_*.h5'):
            stem = f.stem  # round_NNN
            try:
                rounds.append((int(stem.split('_')[1]), f))
            except ValueError:
                continue
        if not rounds:
            continue
        rounds.sort()
        latest_round, latest_path = rounds[-1]
        out.append({
            'slug': run_dir.name,
            'method': _method_from_slug(run_dir.name),
            'latest_round': latest_round,
            'latest_path': str(latest_path),
            'all_rounds': [r for r, _ in rounds],
        })
    return out


def discover_gtp_binaries(gtp_dir: str = 'gtp_agents') -> List[dict]:
    """Detect locally available GTP engines.

    Reports what the registry can already build via auto-discovery
    (gnugo / pachi / katago / fuego / leela).  We only check whether the
    registry's ``find_binary`` resolves the name — actual subprocess
    invocation happens later.
    """
    from alphigo.agent.gtp_agent import find_binary
    out = []
    for name in ('gnugo', 'pachi', 'katago', 'fuego', 'leela'):
        try:
            path = find_binary(name)
        except Exception:
            path = None
        if path:
            out.append({'name': name, 'binary': path})
    _ = gtp_dir  # kept for API symmetry; find_binary already searches it
    return out


def list_all_for_picker() -> List[Tuple[str, str, dict]]:
    """Return [(label, agent_spec, meta)] entries for a numbered picker.

    ``agent_spec`` is a string the registry can parse via parse_agent_spec
    (e.g. "pg:agent_file=runs/.../round_050.h5" or "gnugo").  ``meta`` is
    the underlying discovery dict (so callers can offer round selection).
    """
    items = []
    for run in discover_run_agents():
        method = run['method']
        # Normalise alphazero method slug -> registry name.
        registry_name = 'alphazero' if method == 'alphazero' else method
        spec = f"{registry_name}:agent_file={run['latest_path']}"
        label = (f"{run['slug']} (round {run['latest_round']:03d}, "
                 f"{len(run['all_rounds'])} rounds available)")
        items.append((label, spec, run))
    for gtp in discover_gtp_binaries():
        items.append((f"{gtp['name']} (GTP, {gtp['binary']})",
                      gtp['name'], gtp))
    return items
