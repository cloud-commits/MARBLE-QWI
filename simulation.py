import random
import numpy as np

from env import (MARBLEEnv, define_environment_transition, 
                 define_transition_matrices, define_reward_matrices, 
                 make_user_dynamics)
from agent import QWI_MARBLE, create_state_table
from baseline import TheoreticalBaseline
from helper_functions import (plot_average_rewards, plot_results_index, 
                              save_results, save_metadata, _open_log,
                              plot_aggregate_avg_reward, plot_aggregate_indices,
                              load_results_for_seeds)


def run_policy_comparisons(timesteps=500, gamma=0.95, nUsers=100, nActivate=10, 
                          nStates=4, epsilon=0.05, sync_interval=10, SEED=2025, 
                          heterogeneous=False, trans_conc=400.0, reward_sigma=0.10):
    """
    Run comparison of QWI, Random, and Oracle policies.
    """
    stateTable = np.arange(1, nStates + 1, dtype=np.uint32)
    
    # Define environment
    H = define_environment_transition()
    transitions_base = define_transition_matrices()
    rewards_base = define_reward_matrices()

    log, _logf = _open_log(SEED, timesteps, gamma, epsilon, heterogeneous)
    
    # Save metadata
    save_metadata(SEED, timesteps, gamma, epsilon, nUsers, nActivate, nStates,
                 sync_interval, heterogeneous, trans_conc, reward_sigma)

    # Generate user-specific dynamics
    user_transitions = {}
    user_rewards = {}
    for i in range(nUsers):
        Ti, Ri = make_user_dynamics(i, transitions_base, rewards_base,
                                    seed=SEED, heterogeneous=heterogeneous,
                                    trans_conc=trans_conc, reward_sigma=reward_sigma)
        user_transitions[i] = Ti
        user_rewards[i] = Ri

    # Compute optimal indices
    if not heterogeneous:
        baseline = TheoreticalBaseline(nStates, gamma, 2, transitions_base, 
                                      rewards_base, H)
        optimal_indices, optimal_Q = baseline.compute_optimal_whittle_indices()
        
        # Build averaged model arrays
        P_true = np.zeros((nStates, 2, nStates))
        R_true = np.zeros((nStates, 2))
        for a in range(2):
            P_true[:, a, :] = baseline.avg_transitions[a]
        for s in range(nStates):
            for a in range(2):
                R_true[s, a] = baseline.avg_rewards[(s, a)]
        opt_indices_for_plot = optimal_indices
    else:
        # Per-user models and optimal indices
        P_true_per_user = {}
        R_true_per_user = {}
        optimal_indices_per_user = {}
        
        for i in range(nUsers):
            base_i = TheoreticalBaseline(nStates, gamma, 2, user_transitions[i], 
                                        user_rewards[i], H)
            inds_i, _ = base_i.compute_optimal_whittle_indices()
            optimal_indices_per_user[i] = inds_i

            P_i = np.zeros((nStates, 2, nStates))
            R_i = np.zeros((nStates, 2))
            for a in range(2):
                P_i[:, a, :] = base_i.avg_transitions[a]
            for s in range(nStates):
                for a in range(2):
                    R_i[s, a] = base_i.avg_rewards[(s, a)]
            P_true_per_user[i] = P_i
            R_true_per_user[i] = R_i
        
        opt_indices_for_plot = optimal_indices_per_user[0]

    # Create environments
    envs_qwi = {
        i: MARBLEEnv(SEED + i, gamma, nStates, 2, user_transitions[i], 
                     user_rewards[i], H)
        for i in range(nUsers)
    }
    envs_random = {
        i: MARBLEEnv(SEED + 10_000 + i, gamma, nStates, 2, user_transitions[i], 
                     user_rewards[i], H)
        for i in range(nUsers)
    }
    envs_oracle = {
        i: MARBLEEnv(SEED + 20_000 + i, gamma, nStates, 2, user_transitions[i], 
                     user_rewards[i], H)
        for i in range(nUsers)
    }

    # Create QWI agents
    if not heterogeneous:
        users_qwi = {
            i: QWI_MARBLE(nUsers, gamma, envs_qwi[i], stateTable, 2,
                         P_true=P_true, R_true=R_true)
            for i in range(nUsers)
        }
    else:
        users_qwi = {
            i: QWI_MARBLE(nUsers, gamma, envs_qwi[i], stateTable, 2,
                         P_true=P_true_per_user[i], R_true=R_true_per_user[i])
            for i in range(nUsers)
        }

    # Reset environments
    states_qwi = {i: envs_qwi[i].reset() for i in envs_qwi}
    states_random = {i: envs_random[i].reset() for i in envs_random}
    states_oracle = {i: envs_oracle[i].reset() for i in envs_oracle}

    # Initialize tracking variables
    total_qwi, total_random, total_oracle = 0, 0, 0
    cumulative_qwi, cumulative_random, cumulative_oracle = [], [], []
    avg_qwi, avg_random, avg_oracle = [], [], []
    
    if heterogeneous:
        indices_history_users = {i: {s: [] for s in range(nStates)} 
                                for i in range(nUsers)}
    else:
        indices_history = {s: [] for s in range(nStates)}

    # Main simulation loop
    for t in range(timesteps):
        # QWI Policy
        index = {i: users_qwi[i].indices[states_qwi[i][0]-1] for i in users_qwi}
        if np.random.rand() > epsilon:
            top_qwi = sorted(index, key=index.get, reverse=True)[:nActivate]
        else:
            top_qwi = np.random.choice(list(index.keys()), nActivate, replace=False)
        actions_qwi = [1 if i in top_qwi else 0 for i in range(nUsers)]

        reward_qwi_t = 0
        for i in range(nUsers):
            next_state, r = envs_qwi[i].step(actions_qwi[i])
            users_qwi[i].collect_experience(states_qwi[i], next_state, 
                                           actions_qwi[i], r)
            states_qwi[i] = next_state
            reward_qwi_t += r
        
        # Synchronous update
        if (t + 1) % sync_interval == 0:
            for i in range(nUsers):
                users_qwi[i].synchronous_qTable_update()
                users_qwi[i].update_index(states_qwi[i])
            
        total_qwi += reward_qwi_t
        cumulative_qwi.append(total_qwi)
        avg_qwi.append(total_qwi / (t + 1))
        
        # Track indices
        if heterogeneous:
            for i in range(nUsers):
                for s in range(nStates):
                    indices_history_users[i][s].append(users_qwi[i].indices[s])
        else:
            for s in range(nStates):
                indices_history[s].append(users_qwi[0].indices[s])

        # Oracle Policy
        if not heterogeneous:
            index_oracle = {i: optimal_indices[states_oracle[i][0] - 1] 
                          for i in envs_oracle}
        else:
            index_oracle = {i: optimal_indices_per_user[i][states_oracle[i][0] - 1] 
                          for i in envs_oracle}

        top_oracle = sorted(index_oracle, key=index_oracle.get, reverse=True)[:nActivate]
        actions_oracle = [1 if i in top_oracle else 0 for i in range(nUsers)]

        reward_oracle_t = 0
        for i in range(nUsers):
            next_state, r = envs_oracle[i].step(actions_oracle[i])
            states_oracle[i] = next_state
            reward_oracle_t += r

        total_oracle += reward_oracle_t
        cumulative_oracle.append(total_oracle)
        avg_oracle.append(total_oracle / (t + 1))

        # Random Policy
        random_users = np.random.choice(range(nUsers), nActivate, replace=False)
        actions_random = [1 if i in random_users else 0 for i in range(nUsers)]

        reward_random_t = 0
        for i in range(nUsers):
            next_state, r = envs_random[i].step(actions_random[i])
            states_random[i] = next_state
            reward_random_t += r
            
        total_random += reward_random_t
        cumulative_random.append(total_random)
        avg_random.append(total_random / (t + 1))

        # Logging
        if (t + 1) % 1000 == 0:
            log(f"\nTimestep {t+1}, Setting [ seed {SEED}, gamma {gamma}, "
                f"epsilon {epsilon}, itr {timesteps} ]:")
            log(f"  QWI Total: {total_qwi:.1f}, Random Total: {total_random:.1f}")
            log(f"  Convergence:")
            log(f"{'State':<10} {'Optimal':<15} {'Learned':<15} "
                f"{'Error':<15} {'Error(%)':<15}")

            for s in range(nStates):
                learned = users_qwi[0].indices[s]
                theoretical = opt_indices_for_plot[s]
                error = abs(learned - theoretical)
                den = abs(theoretical)
                error_r = 100*error/den if den > 1e-12 else 100*error/1e-12
                log(f"{s+1:<10} {theoretical:<15.6f} {learned:<15.6f} "
                    f"{error:<15.6f} {error_r:<15.6f}")

    # Prepare results
    results = {
        "avg_qwi": avg_qwi,
        "avg_random": avg_random,
        "avg_oracle": avg_oracle,
        "cumulative_qwi": cumulative_qwi,
        "cumulative_random": cumulative_random,
        "cumulative_oracle": cumulative_oracle,
        "indices_history": (indices_history if not heterogeneous 
                          else indices_history_users),
        "optimal_indices": (optimal_indices if not heterogeneous 
                          else optimal_indices_per_user),
        "heterogeneous": heterogeneous
    }

    if heterogeneous:
        results["trans_conc"] = trans_conc
        results["reward_sigma"] = reward_sigma

    # Save results
    save_results(results, SEED)

    # Generate plots
    if heterogeneous:
        plot_results_index(avg_qwi, avg_random, cumulative_qwi, cumulative_random,
                          indices_history_users, optimal_indices_per_user,
                          nStates, gamma, epsilon, timesteps, SEED, 
                          heterogeneous=True)
    else:
        plot_results_index(avg_qwi, avg_random, cumulative_qwi, cumulative_random,
                          indices_history, optimal_indices,
                          nStates, gamma, epsilon, timesteps, SEED, 
                          heterogeneous=False)

    plot_average_rewards(avg_qwi, avg_random, avg_oracle, gamma, epsilon, 
                        timesteps, SEED, heterogeneous=heterogeneous)

    try:
        _logf.close()
    except Exception:
        pass

    return results


def run_many_seeds_and_plot_aggregate(seeds, *, timesteps, gamma, epsilon, 
                                     heterogeneous, nUsers=100, nActivate=10, 
                                     nStates=4, sync_interval=10, 
                                     trans_conc=400.0, reward_sigma=0.10):
    """
    Run your pipeline for several seeds and plot mean±std of average rewards.
    Returns the list of per-seed results dicts.
    """
    results_all = []
    
    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)
        
        res = run_policy_comparisons(
            timesteps=timesteps, 
            gamma=gamma,
            nUsers=nUsers, 
            nActivate=nActivate, 
            nStates=nStates,
            epsilon=epsilon, 
            sync_interval=sync_interval,
            SEED=sd, 
            heterogeneous=heterogeneous,
            trans_conc=trans_conc, 
            reward_sigma=reward_sigma
        )
        results_all.append(res)

    # Collect curves
    avg_qwi_list = [r["avg_qwi"] for r in results_all]
    avg_rand_list = [r["avg_random"] for r in results_all]
    avg_oracle_list = [r["avg_oracle"] for r in results_all]

    # Plot aggregates
    plot_aggregate_avg_reward(avg_qwi_list, avg_rand_list, avg_oracle_list,
                              timesteps, gamma, epsilon, heterogeneous, seeds)

    plot_aggregate_indices(results_all, nStates, timesteps, gamma, epsilon, 
                          heterogeneous, seeds)

    return results_all


def replot_from_saved(seeds, *, timesteps, gamma, epsilon, heterogeneous, nStates):
    """Recreate aggregate plots later without rerunning simulations."""
    results_all = load_results_for_seeds(seeds)
    
    avg_qwi_list = [r["avg_qwi"] for r in results_all]
    avg_rand_list = [r["avg_random"] for r in results_all]
    avg_oracle_list = [r["avg_oracle"] for r in results_all]
    
    plot_aggregate_avg_reward(avg_qwi_list, avg_rand_list, avg_oracle_list,
                              timesteps, gamma, epsilon, heterogeneous, seeds)
    
    plot_aggregate_indices(results_all, nStates, timesteps, gamma, epsilon, 
                          heterogeneous, seeds)
    
    return results_all