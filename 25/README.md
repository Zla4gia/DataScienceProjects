# 25 - Build an AI Agent to Master a Game
**Topic:** AI Agents  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2025/09/09/build-an-ai-agent-to-master-a-game-using-python/)

**Topic:** AI Agents (reinforcement learning)  |  **Reference:** "Build an AI Agent to Master a Game using Python" (amanxai.com, Sep 2025)

**Question.** Can an agent learn, by trial and error, to balance a pole on a moving cart, and is "deep" reinforcement learning actually needed?

**Reference method:** a Deep Q-Network (DQN) in PyTorch trained on Gymnasium's CartPole-v1, shown by a learning curve and a rendered episode.
**What I add:** (1) the **reward and failure rules** of the game stated and tested; (2) four agents compared on the same footing: a **random** policy, a **hand-written rule** with no learning, **tabular Q-learning** on a discretised state and the **DQN**; (3) **several random seeds** (RL results vary a lot between runs); (4) **how stable** training is (peak vs final performance); (5) **robustness**: what happens when gravity, pole length or the force of the push change after training; (6) sample efficiency (how many game steps each learner needs).

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Method | one DQN, one learning curve, a rendered episode | random policy, a one-line rule, tabular Q-learning, DQN; **3 seeds each** |
| Benchmark | none | random 23 steps; **hand-written rule with no learning: 500 in 100/100 episodes** |
| Result | the DQN "masters" the game | DQN 500 on 3/3 seeds, Q-learning 500 / 218 / 486; DQN learns faster (first success at 20-44k steps vs 55-85k) but both are unstable during training |
| Robustness | not tested | 6 modified worlds: all learned policies and the rule at 491-500, so the test did not discriminate |
| Saved | - | DQN weights, Q-table, GIFs of a random and a trained agent |

**What I learned.** (1) Compare against the simplest thing that could work: a one-line rule is as good as either learner here. (2) Report seeds, not a single run: Q-learning had one bad seed. (3) Performance during training oscillates, so the "final" score depends on when you stop; keep a checkpoint chosen on separate episodes. (4) A robustness test must be able to fail: this one could not separate the policies.

**Limits.** One game with four state variables; 3 seeds; 10-episode periodic tests are noisy; hyper-parameters were not tuned (a different bin layout or learning rate could change Q-learning a lot); wall-clock times were measured on a busy shared CPU and are not reported. **Next step:** keep the best checkpoint on separate validation episodes, tune the Q-table bins, and try a harder environment (LunarLander) where a hand rule is not available.

---

## Files
* `25 AI Agent to Master a Game.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `cartpole_dqn.gif`, `cartpole_dqn.pt`, `cartpole_qtable.npy`, `cartpole_random.gif`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "25 AI Agent to Master a Game.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
