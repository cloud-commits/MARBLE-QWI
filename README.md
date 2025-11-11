# MARBLE: Multi-Armed Restless Bandits in Latent Environments

This repository contains the implementation used in our experiments for **MARBLE-QWI**, a reinforcement learning algorithm designed to learn **Whittle indices** in **Multi-Armed Restless Bandits in latent environment setting**.

To run the simulation:

```bash
python main.py
```

- `env.py`**: Simulates user state evolution in changing environmental contexts
- `agent.py`**: Q-learning agent that learns Whittle indices through experience
- `baseline.py`**: Computes optimal solutions using full knowledge of dynamics
- `helper_functions.py`: Utility functions for plotting and data management
- `simulation.py`: Functions for running experiments and managing multi-seed runs

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

## Arguments

| **Argument**      | **Type** | **Description**                                                                      |
|-------------------|----------|--------------------------------------------------------------------------------------|
| `seed`            | `int`    | Random seed to run the simulation.                                                   |
| `timesteps`       | `int`    | Number of timesteps per simulation run. Determines the training duration.            |
| `gamma`           | `float`  | Discount factor for future rewards (0 < γ < 1).                                      |
| `nUsers`          | `int`    | Total number of simulated users (arms).                                              |
| `nActivate`       | `int`    | Number of users activated per timestep.                                              |
| `nStates`         | `int`    | Number of discrete user states in the model.                                         |
| `epsilon`         | `float`  | Exploration probability for the epsilon-greedy policy.       |
| `sync_interval`   | `int`    | Frequency of synchronous Q-table updates (in timesteps).                             |
| `heterogeneous`   | `bool`   | Whether to use heterogeneous user dynamics across arms.                              |
| `trans_conc`      | `float`  | Dirichlet concentration parameter for transition perturbations (heterogeneous mode). |
| `reward_sigma`    | `float`  | Lognormal sigma for reward perturbations (heterogeneous mode).                       |

## Example Usage

### Single Seed Run

```python
from simulation import run_policy_comparisons

results = run_policy_comparisons(
    timesteps=500_000,
    gamma=0.8,
    epsilon=0.1,
    SEED=2025,
    heterogeneous=True
)
```

### Multi-Seed Analysis

```python
from simulation import run_many_seeds_and_plot_aggregate

seeds = [2025, 2026, 2027, 2028, 2029]
results = run_many_seeds_and_plot_aggregate(
    seeds,
    timesteps=500_000,
    gamma=0.8,
    epsilon=0.1,
    heterogeneous=True
)
```

### Replotting from Saved Results

```python
from simulation import replot_from_saved

seeds = [2025, 2026, 2027, 2028, 2029]
replot_from_saved(
    seeds,
    timesteps=500_000,
    gamma=0.8,
    epsilon=0.1,
    heterogeneous=True,
    nStates=4
)
```

## Output

Results are saved in the `Results/` directory:
- `.pkl` files: Results for each seed
- `.png` files: Plots for average rewards and Whittle index convergence
- `.log` files: Detailed execution logs with convergence metrics
- `.json` files: Run metadata and configuration

## Algorithm Details

### MARBLE-QWI Agent
- Uses Q-learning to estimate Whittle indices online
- Synchronous Q-table updates every `sync_interval` steps
- Adaptive learning rates: $\alpha(n)$ and $\beta(n)$
- $\epsilon$-greedy policy for exploration

### MARBLE Environment
- Two latent environments: **E1 (Good)** and **E2 (Bad)**
- 4 user engagement states
- 2 actions: **0 (passive)**, **1 (active/send recommendation)**
- Environment dynamics follow a Markov chain

### Baselines
- **Random**: Randomly selects arms to activate at each timestep
- **Oracle**: Uses theoretically optimal Whittle indices (requires full knowledge of dynamics)
