# scripts/

## Gumbel AlphaZero experiments (9×9)

| Experiment | Runner | Results |
|---|---|---|
| Exp. 1 | `run_gumbel_loop_b9.sh` (initial configuration; later phases were changed by hand, see the paper) | `results/gumbel_loop_b9_v1.csv` |
| Exp. 2 | `run_gumbel_loop_b9_v3.sh` | `results/gumbel_loop_b9_v3.csv` |
| Exp. 3 | `run_gumbel_loop_b9_v9.sh` | `results/gumbel_loop_b9_v9.csv` |
| Exp. 4 | `run_gumbel_loop_b9_v10.sh` | `results/gumbel_loop_b9_v10.csv` |
| Exp. 5 | `run_gumbel_loop_b9_v11.sh` | `results/gumbel_loop_b9_v11.csv` |
| Exp. 6 | `run_gumbel_loop_b9_v12.sh` | `results/gumbel_loop_b9_v12.csv` |
| Exp. 7 | `run_gumbel_loop_b9_v12b.sh` | `results/gumbel_loop_b9_v12b.csv` |
| Exp. 8 | `run_gumbel_loop_b9_v12c.sh` | `results/gumbel_loop_b9_v12c.csv` |
| Re-run with Exp. 6 settings (source of `model/round_019_agent.h5`) | `run_gumbel_demo.sh` | `results/gumbel_demo.csv` |

The runner scripts are in their final state. Settings that were changed while a run was in progress are described in the paper. Intermediate loop versions that are not part of the reported experiments are not included.

Components used by the runners:

| Script | Role |
|---|---|
| `generate_selfplay_data_mp.py` | Multiprocess self-play generation: GPU inference server (ONNX Runtime / TensorRT) and CPU workers |
| `train_distill_b19.py` | Training of one round (policy, value, ownership and score heads) |
| `quick_eval_mcts.py` | Games against GNU Go and Pachi |
| `head_to_head.py` | Games between two agents |
| `probe_policy_value.py` | Policy cross-entropy and value accuracy on a fixed set of positions |
| `_test_tactics.py` | 25-position tactics suite |
| `generate_seeded_openings.py`, `seeded_openings_edge_v3.json` | Seeded openings used in Exp. 8 |
| `plot_gumbel_loop.py` | Plots of the per-round results |
| `puct_inversion_exp.py` | Sign experiment (CPU) |

### Result columns

- `policy_ce`, `value_acc`: policy cross-entropy and value-sign accuracy on 10,000 fixed positions from human OGS games (a transfer metric, not the training distribution).
- `self_vacc`: value-sign accuracy on held-out self-play positions of the same round.
- `gnugo_wins`, `gnugo_ph_wins`: wins out of 20 games against GNU Go level 1, without and with a pass heuristic.
- `h2hprev_wins`: wins out of 20 games against the checkpoint of the same run 10 rounds earlier (in Exp. 1 up to round 40: 5 rounds earlier). In Exp. 7 and 8, round 0 is a copy of Exp. 6 round 80.
- `h2h0_wins`: wins against round 0. The value 10/20 in round 0 is written by the script, not played. In `gumbel_loop_b9_v12.csv` (Exp. 6) these two columns hold instead a 40-game match against Exp. 5 round 132 (rounds 20, 40, 60 and 80).
- `yard_wins` (Exp. 7, 8): match against the starting weights.
- `pachi_wins`: wins out of 20 games against Pachi. In `gumbel_loop_b9_v12.csv` this is the unnamed last column (8000 playouts in rounds 50 to 60, 16,000 from round 62).
- `positions`, `gen_secs`: self-play positions and generation time of the round.

## Supervised learning and PUCT experiments (thesis chapters 4 and 5)

| Script | Role |
|---|---|
| `thesis_run.py` (repository root) | Supervised training and evaluation across encoders and networks |
| `sl_sweep_9x9.sh` | 9×9 supervised sweep |
| `encode_ogs_supervised.py`, `run_ogs_train*.sh` | OGS 9×9 dataset and supervised training |
| `sims_sweep.sh`, `bench_az_vs_gnugo_mp.py` | Win rate as a function of the number of simulations |
| `mcts_correctness_b19.sh` | Same sweep with different value sources (network, zero, rollout) |
| `seed_variance_chain.sh` | Seed variance of the value-head retraining |
| `run_separate_value_experiments.sh`, `sims_800_*.sh` | Separate value network |
| `run_selfplay_value_experiment.sh` | Value network trained on self-play |
| `run_ownership_eval.sh` | Ownership head as a value estimate |
| `diagnose_positions.py`, `run_diagnose_*.sh` | Comparison of the value heads on positions from real games |
| `run_9x9_selfplay_loop.sh` | 9×9 PUCT self-play loop |
| `run_postfix_rerun.sh` | 9×9 re-evaluation after the PUCT sign fix |

Most PUCT results in chapters 4 and 5 were measured before the sign fix in `alphigo/agent/zero_agent.py` (see the paper). Results with 1 simulation do not depend on the sign.
