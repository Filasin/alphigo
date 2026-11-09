"""Agent registry: create any agent by name + keyword config.

Usage:
    from alphigo.agent.registry import create_agent, list_agents

    bot = create_agent("random")
    bot = create_agent("mcts", num_rounds=500, temperature=1.4)
    bot = create_agent("dl", agent_file="path/to/agent.h5")
"""

from alphigo.agent.base import Agent
from alphigo.agent.naive import RandomBot


_REGISTRY = {}


def register(name, *, aliases=(), help_text=""):
    """Decorator that registers an agent builder function."""
    def decorator(fn):
        entry = {"builder": fn, "help": help_text}
        _REGISTRY[name] = entry
        for alias in aliases:
            _REGISTRY[alias] = entry
        return fn
    return decorator


def create_agent(name, **kwargs):
    """Instantiate an agent by its registered name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(set(
            k for k, v in _REGISTRY.items()
            if not any(k2 != k and _REGISTRY[k2] is v for k2 in _REGISTRY)
        ) | set(_REGISTRY.keys())))
        raise ValueError(
            f"Unknown agent '{name}'. Available: {available}"
        )
    return _REGISTRY[name]["builder"](**kwargs)


def list_agents():
    """Return list of (canonical_name, help_text) for unique agents."""
    seen = set()
    result = []
    for name, entry in _REGISTRY.items():
        builder_id = id(entry["builder"])
        if builder_id not in seen:
            seen.add(builder_id)
            result.append((name, entry["help"]))
    return result


# ── Built-in agent builders ──────────────────────────────────────────

@register("random", aliases=("rand",),
          help_text="Random legal moves, avoids filling own eyes")
def _build_random(**_kwargs):
    return RandomBot()


@register("mcts", aliases=("monte-carlo",),
          help_text="Monte-Carlo Tree Search (num_rounds, temperature)")
def _build_mcts(num_rounds=500, temperature=1.4, **_kwargs):
    from alphigo.mcts.monte_carlo import MCTSAgent
    return MCTSAgent(int(num_rounds), float(temperature))


@register("dl", aliases=("deep", "deeplearning"),
          help_text="Deep-learning policy net (agent_file=<path>, temperature=0.0)")
def _build_dl(agent_file=None, temperature=0.0, **_kwargs):
    if agent_file is None:
        raise ValueError("dl agent requires agent_file=<path to .h5>")
    import h5py
    from alphigo.agent.deep_learning import load_prediction_agent
    with h5py.File(agent_file, "r") as h5:
        agent = load_prediction_agent(h5)
    agent.temperature = float(temperature)
    return agent


@register("katago",
          help_text="KataGo GTP engine (model=<path>, max_visits=100, binary=<path>)")
def _build_katago(model=None, binary=None, max_visits=100, config=None,
                  **_kwargs):
    if model is None:
        raise ValueError(
            "katago agent requires model=<path to weights file>. "
            "Download weights from https://katagotraining.org/networks/")
    from alphigo.agent.gtp_agent import GtpAgent, find_binary
    binary = binary or find_binary('katago')
    cmd = [binary, 'gtp', '-model', model]
    if config:
        cmd.extend(['-config', config])
    cmd.extend(['-override-config', f'maxVisits={int(max_visits)}'])
    return GtpAgent(cmd)


@register("gumbel",
          help_text="Gumbel AlphaZero search "
                    "(agent_file=<path>, num_simulations=32, m=16)")
def _build_gumbel(agent_file=None, num_simulations=32, m=16,
                  c_visit=50.0, c_scale=1.0, batch_size=16, **_kwargs):
    if agent_file is None:
        raise ValueError("gumbel agent requires agent_file=<path to .h5>")
    import h5py
    from alphigo.agent.gumbel_zero_agent import load_gumbel_agent
    with h5py.File(agent_file, "r") as h5:
        return load_gumbel_agent(
            h5, num_simulations=int(num_simulations), m=int(m),
            c_visit=float(c_visit), c_scale=float(c_scale),
            batch_size=int(batch_size))


@register("gnugo",
          help_text="GNU Go GTP engine (level=10, binary=<path>)")
def _build_gnugo(binary=None, level=10, **_kwargs):
    from pathlib import Path
    from alphigo.agent.gtp_agent import GtpAgent, find_binary
    binary = binary or find_binary('gnugo')
    cwd = str(Path(binary).parent)
    cmd = [binary, '--mode', 'gtp', '--level', str(int(level))]
    return GtpAgent(cmd, cwd=cwd)


@register("pachi",
          help_text="Pachi GTP engine (playouts=1000, threads=1, binary=<path>)")
def _build_pachi(binary=None, playouts=1000, threads=1, **_kwargs):
    from pathlib import Path
    from alphigo.agent.gtp_agent import GtpAgent, find_binary
    binary = binary or find_binary('pachi')
    cwd = str(Path(binary).parent)
    effective = max(int(playouts), 500)  # Pachi 12.x requires >= 500 playouts
    if effective != int(playouts):
        import warnings
        warnings.warn(f'pachi: playouts={playouts} below minimum 500, using 500')
    # -t =N: fixed simulations per move; threads=N is a uct engine arg.
    cmd = [binary]
    # --nojosekifix: disable josekifix module which tries to launch KataGo.
    # Only available when Pachi is built with josekifix support.
    if _pachi_supports_nojosekifix(binary):
        cmd.append('--nojosekifix')
    cmd.extend(['-t', f'={effective}', f'threads={int(threads)}'])
    return GtpAgent(cmd, cwd=cwd)


def _pachi_supports_nojosekifix(binary):
    """Check if this Pachi binary was built with josekifix (and thus needs --nojosekifix)."""
    import subprocess
    try:
        out = subprocess.run(
            [binary, '--help'], capture_output=True, text=True, timeout=5,
        )
        return '--nojosekifix' in (out.stdout + out.stderr)
    except Exception:
        return False


@register("fuego",
          help_text="Fuego GTP engine (max_games=5000, binary=<path>)")
def _build_fuego(binary=None, max_games=5000, **_kwargs):
    from pathlib import Path
    from alphigo.agent.gtp_agent import GtpAgent, find_binary
    binary = binary or find_binary('fuego')
    cwd = str(Path(binary).parent)
    cmd = [binary,
           '--config', '',  # suppress default config file
           f'uct_param_player max_games={int(max_games)}']
    return GtpAgent(cmd, cwd=cwd)


@register("leela",
          help_text="Leela GTP engine (playouts=1600, binary=<path>)")
def _build_leela(binary=None, playouts=1600, **_kwargs):
    from pathlib import Path
    from alphigo.agent.gtp_agent import GtpAgent, find_binary
    binary = binary or find_binary('leela')
    cwd = str(Path(binary).parent)
    cmd = [binary, '--gtp', '--playouts', str(int(playouts)), '--noponder']
    return GtpAgent(cmd, cwd=cwd)


@register("pg", aliases=("policy", "policy-gradient"),
          help_text="Policy gradient RL agent (agent_file=<path>, temperature=0.0)")
def _build_pg(agent_file=None, temperature=0.0, **_kwargs):
    if agent_file is None:
        raise ValueError("pg agent requires agent_file=<path to .h5>")
    from alphigo.training.method import get_method
    ag = get_method('pg').load_agent(agent_file)
    ag.set_temperature(float(temperature))
    return ag


@register("q", aliases=("q-book", "q-learning"),
          help_text="Q-learning RL agent, book-style two-input "
                    "(agent_file=<path>, epsilon=0.0)")
def _build_q(agent_file=None, epsilon=0.0, **_kwargs):
    if agent_file is None:
        raise ValueError("q agent requires agent_file=<path to .h5>")
    from alphigo.training.method import get_method
    ag = get_method('q').load_agent(agent_file)
    ag.set_epsilon(float(epsilon))
    return ag


@register("ac", aliases=("actor-critic", "actor_critic"),
          help_text="Actor-critic RL agent "
                    "(agent_file=<path>, epsilon=0.0)")
def _build_ac(agent_file=None, epsilon=0.0, **_kwargs):
    if agent_file is None:
        raise ValueError("ac agent requires agent_file=<path to .h5>")
    from alphigo.training.method import get_method
    ag = get_method('ac').load_agent(agent_file)
    ag.set_epsilon(float(epsilon))
    return ag


@register("alphazero", aliases=("zero", "az"),
          help_text="AlphaZero PUCT+two-head agent "
                    "(agent_file=<path>, num_simulations=50, c_puct=1.5, "
                    "temperature=0.0, value_mode=network|zero|rollout|ownership, "
                    "rollout_depth=80)")
def _build_alphazero(agent_file=None, num_simulations=None, c_puct=None,
                     temperature=0.0, value_mode='network',
                     rollout_depth=80, **_kwargs):
    if agent_file is None:
        raise ValueError("alphazero agent requires agent_file=<path to .h5>")
    import h5py
    from alphigo.agent.zero_agent import load_zero_agent
    with h5py.File(agent_file, "r") as h5:
        agent = load_zero_agent(
            h5,
            num_simulations=num_simulations,
            c_puct=c_puct,
            value_mode=value_mode,
            rollout_depth=int(rollout_depth),
        )
    agent.set_temperature(float(temperature))
    return agent


@register("gtp",
          help_text="Generic GTP engine (cmd=<full command string>)")
def _build_gtp(cmd=None, **_kwargs):
    if cmd is None:
        raise ValueError(
            "gtp agent requires cmd=<command>  e.g. cmd=gnugo --mode gtp")
    import shlex
    from alphigo.agent.gtp_agent import GtpAgent
    return GtpAgent(shlex.split(cmd))
