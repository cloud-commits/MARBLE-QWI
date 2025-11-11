
import random
import numpy as np
from simulation import run_many_seeds_and_plot_aggregate, replot_from_saved


def main():

    # Simulation parameters
    gamma = 0.8
    epsilon = 0.1
    timesteps = 500_000
    heterogeneous = True
    
    # Environment parameters
    nUsers = 100
    nActivate = 10
    nStates = 4
    sync_interval = 10
    
    # Heterogeneity parameters
    trans_conc = 400.0    # Higher = less variance in transitions
    reward_sigma = 0.10   # Lognormal sigma for reward perturbations
    
    # Random seeds
    seeds = [2025]  # Add more seeds for aggregate analysis
    
    print("=" * 80)
    print("Q-Learning Whittle Index (QWI) Simulation")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Gamma: {gamma}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Heterogeneous: {heterogeneous}")
    print(f"  Number of Users: {nUsers}")
    print(f"  Number to Activate: {nActivate}")
    print(f"  Number of States: {nStates}")
    print(f"  Sync Interval: {sync_interval}")
    print(f"  Seeds: {seeds}")
    if heterogeneous:
        print(f"  Transition Concentration: {trans_conc}")
        print(f"  Reward Sigma: {reward_sigma}")
    print("\n" + "=" * 80)
    print("Starting simulation...\n")
    
    # Run simulation
    results = run_many_seeds_and_plot_aggregate(
        seeds,
        timesteps=timesteps,
        gamma=gamma,
        epsilon=epsilon,
        heterogeneous=heterogeneous,
        nUsers=nUsers,
        nActivate=nActivate,
        nStates=nStates,
        sync_interval=sync_interval,
        trans_conc=trans_conc,
        reward_sigma=reward_sigma
    )
    
    print("\n" + "=" * 80)
    print("Simulation complete!")
    print("Results saved in ./Results/ directory")
    print("=" * 80)
    
    return results


def replot_example():
    """
    Example of replotting from saved results.
    """
    seeds = [2025]
    gamma = 0.8
    epsilon = 0.1
    timesteps = 500_000
    heterogeneous = True
    nStates = 4
    
    results = replot_from_saved(
        seeds,
        timesteps=timesteps,
        gamma=gamma,
        epsilon=epsilon,
        heterogeneous=heterogeneous,
        nStates=nStates
    )
    
    return results


if __name__ == "__main__":
    main()