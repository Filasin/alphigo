# AlphiGo command reference

```bash
# Install dependencies
pip install -r requirements.txt

# Compile the C++ MCTS data generator (Windows MSVC)
cl /O2 /std:c++17 /EHsc generate_mcst_games.cpp /Fe:generate_mcst_games.exe

# Compile the C++ MCTS data generator (Linux/WSL)
g++ -O3 -std=c++17 -pthread -o generate_mcst_games generate_mcst_games.cpp

# Generate self-play data (C++, fast)
./generate_mcst_games.exe --out-dir generated_games/mcst_9x9 --games 1000 --board 9

# After generation, merge batches into train/test splits
python generated_games/merge_batches.py generated_games/mcst_9x9 --tag mcts

# Train a model from NPY files
python train.py --features-train generated_games/mcst_9x9/mcts_oneplane_features_train.npy \
                --labels-train    generated_games/mcst_9x9/mcts_labels_train.npy \
                --features-test   generated_games/mcst_9x9/mcts_oneplane_features_test.npy \
                --labels-test     generated_games/mcst_9x9/mcts_labels_test.npy \
                --encoder oneplane --network small --board-size 9 --epochs 15

# Evaluate checkpoints against GTP engines, print Elo table
python eval_checkpoints.py checkpoints/*.h5 --opponents gnugo pachi:playouts=1000

# Run a headless benchmark (win-rate statistics)
python run.py benchmark --black dl:agent_file=checkpoints/model.h5 --white gnugo -n 50

# Single game with move-by-move board output
python run.py benchmark --black dl:agent_file=checkpoints/model.h5 --white gnugo --moves

# Round-robin tournament
python run.py tournament --agents dl:agent_file=checkpoints/model.h5 gnugo mcts:num_rounds=500

# GUI: human vs bot
python run.py play --black human --white dl:agent_file=checkpoints/model.h5

# End-to-end experiment (train all encoder×network combos → eval → tournament)
python experiment.py --board-size 9 --epochs 15

# ── Reinforcement learning (policy gradient) ──

# 1. Create a fresh (untrained) PolicyAgent
python init_pg_agent.py --agent-out agents/pg_agent.h5 --board-size 9 --encoder oneplane --network small

# 2. Generate self-play experience data
python self_play.py --learning-agent agents/pg_agent.h5 --experience-out experience/exp_0001.h5 \
                    --num-games 500 --board-size 9 --epsilon 0.0

# 3. Train the agent from experience (REINFORCE / policy gradient)
python train_pg.py --learning-agent agents/pg_agent.h5 --agent-out agents/pg_agent_v2.h5 \
                   --lr 0.0001 --bs 512 experience/exp_0001.h5

# 4. Evaluate: pit updated agent against previous version
python run.py benchmark --black pg:agent_file=agents/pg_agent_v2.h5 \
                        --white pg:agent_file=agents/pg_agent.h5 -n 100

# Repeat steps 2-4 with the updated agent to continue the RL cycle.

# ── Automated RL loop (recommended) ──

# Automated: self-play → train → evaluate, repeated for N rounds
python train_rl.py --board-size 19 --encoder oneplane --network small --rounds 10 --games 500

# Bootstrap from a supervised checkpoint (recommended for 19x19)
python train_rl.py --board-size 19 --encoder oneplane --network small --rounds 10 \
                   --bootstrap checkpoints/supervised_oneplane_small_adam_b19_model_epoch_15.h5

# 9x9 RL from scratch
python train_rl.py --board-size 9 --encoder oneplane --network small --rounds 20 --games 500

# With external evaluation opponents
python train_rl.py --board-size 9 --encoder oneplane --network small --rounds 10 \
                   --eval-opponents gnugo pachi:playouts=500

# Via thesis runner (runs all encoder/network combos)
python thesis_run.py --board-size 9 --training-method pg --phase train \
                     --encoders oneplane --networks small --rl-rounds 10 --rl-games 500
```
