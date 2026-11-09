"""PUCT with correct vs inverted sign at a given number of simulations,
against the same network without search (1 simulation).

Usage (from the repository root):
    python scripts/puct_inversion_exp.py . correct  100 40
    python scripts/puct_inversion_exp.py . inverted 100 40
Arguments: <code_root> <correct|inverted> <simulations> <games> [checkpoint]
The checkpoint defaults to model/round_019_agent.h5. Results: results/sign_experiment.csv.
Each game starts with 4 random moves (seed 1000 + game index), then greedy
move selection (T = 0); colors alternate. Runs on CPU only.
"""
import os, sys, math
os.environ['CUDA_VISIBLE_DEVICES']='-1'; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
repo, variant, sims, ngames = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ck = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'round_019_agent.h5')
sys.path.insert(0, os.path.abspath(repo))
import h5py, numpy as np, random
from alphigo.agent.zero_agent import ZeroAgent, load_zero_agent
from alphigo.alphigo_board import GameState, Move
from alphigo.alphigo_types import Player
from alphigo.simulate import simulate_game
if variant == 'inverted':   # exact selection rule before the sign fix
    def _sel(self, node):
        sq = self._c_puct * math.sqrt(node.N); best=None; bs=-1e18
        for c in node.children:
            s = c.Q + sq * c.prior / (1 + c.N)
            if s > bs: bs, best = s, c
        return best
    ZeroAgent._select_child_puct = _sel
def mk(n):
    with h5py.File(ck,'r') as h5: a = load_zero_agent(h5, num_simulations=n, c_puct=1.5, batch_size=1)
    a.set_temperature(0.0); return a
agent, opp = mk(sims), mk(1)
class Opening:
    """Plays k random legal moves first (seeded), then delegates."""
    def __init__(s, inner, rng): s.inner, s.rng = inner, rng
    def select_move(s, gs):
        if len(gs.move_history()) < 4:
            cands=[m for m in gs.legal_moves() if m.point is not None]
            return s.rng.choice(cands)
        return s.inner.select_move(gs)
wins=0
for g in range(ngames):
    rng = random.Random(1000+g)
    a, o = Opening(agent, rng), Opening(opp, rng)
    col = Player.black if g%2==0 else Player.white
    r = simulate_game(a, o, 9) if col==Player.black else simulate_game(o, a, 9)
    wins += (r['winner']==col)
print(f'RESULT variant={variant} sims={sims} wins={wins}/{ngames}', flush=True)
