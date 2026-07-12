# src/ppo — the CHRL-lineage PPO stack

The reinforcement-learning training and environment code of the predecessor line (paper §2): the
constrained-hierarchical PPO trader, its weight-based environments, teachers, and critics. The
CRYSTAL-1 interpretability work reads this stack's frozen outputs; you rarely retrain here.

| Group | Modules |
|---|---|
| Policies | `dirichlet_policy.py` (simplex weights), `w1_budget_trader_policy.py`, `instrumented_ppo.py` (SB3 subclass that logs rollout diagnostics), `q_critic.py` |
| Environments | `stage0_1_weight_env.py`, `pm_trader_env.py`, `two_agent_env.py`, `w1_budget_trader_env.py`, `synthetic_lob_execution_env.py`, `weight_panel.py` |
| Trainers | `stage0_1_train.py`, `stage0_1_two_agent_{,ctde_,joint_}train.py`, `stage0_1_w1_{,vectorized_}train.py`, `stage0_1_h1_pm_trader_train.py`, `stage0_1_synthetic_lob_train.py` |
| Rewards / helpers | `stage0_1_rewards.py`, `pretrain_teachers.py`, `vec_env_factory.py`, `w1_*` config/risk helpers |

**Heads-up:** several files here are large training monoliths (2000–4000 LOC) on the
un-runnable-without-data/GPU path. They carry good module docstrings but are not split, by design:
a blind refactor cannot be verified without a full retrain, which the lab's "verify or don't
touch" rule forbids. Read the module docstring, not the whole file.
