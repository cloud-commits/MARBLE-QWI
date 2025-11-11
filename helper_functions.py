import os
import json
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_OUT_DIR = os.path.join(_BASE_DIR, "Results")
os.makedirs(_OUT_DIR, exist_ok=True)


def _savefig(fname, **kwargs):
    """Save figure to Results directory."""
    path = os.path.join(_OUT_DIR, fname)
    plt.savefig(path, **kwargs)
    return path


def _run_tag(seed, timesteps, gamma, epsilon, heterogeneous):
    """Generate a unique tag for a simulation run."""
    het = "heteroTrue" if heterogeneous else "heteroFalse"
    return f"sd{seed}_itr{timesteps}_gamma{gamma}_eps{epsilon}_{het}"


def _open_log(seed, timesteps, gamma, epsilon, heterogeneous):
    """
    Open a log file for this run.
    
    Returns
    -------
    log : function
        Logging function that prints to console and file
    logf : file
        Log file handle
    """
    tag = _run_tag(seed, timesteps, gamma, epsilon, heterogeneous)
    path = os.path.join(_OUT_DIR, f"run_{tag}.log")
    logf = open(path, "w", buffering=1)
    
    def log(msg=""):
        print(msg)
        print(msg, file=logf)
    
    return log, logf


def save_metadata(seed, timesteps, gamma, epsilon, nUsers, nActivate, nStates,
                 sync_interval, heterogeneous, trans_conc, reward_sigma):
    """Save run metadata as JSON."""
    meta = {
        "seed": seed,
        "timesteps": timesteps,
        "gamma": gamma,
        "epsilon": epsilon,
        "nUsers": nUsers,
        "nActivate": nActivate,
        "nStates": nStates,
        "sync_interval": sync_interval,
        "heterogeneous": heterogeneous,
        "trans_conc": trans_conc,
        "reward_sigma": reward_sigma,
        "timestamp": datetime.now().isoformat()
    }
    
    fname = f"meta_{_run_tag(seed, timesteps, gamma, epsilon, heterogeneous)}.json"
    path = os.path.join(_OUT_DIR, fname)
    with open(path, "w") as mf:
        json.dump(meta, mf, indent=2)


def load_results_for_seeds(seeds):
    """
    Load previously saved .pkl results.
    
    Parameters
    ----------
    seeds : list
        List of seed values
        
    Returns
    -------
    results : list
        List of result dictionaries
    """
    out = []
    for sd in seeds:
        p = os.path.join(_OUT_DIR, f"results_whittle_sync_seed{sd}.pkl")
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def save_results(results, seed):
    """
    Save results dictionary to pickle file.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    seed : int
        Seed value for filename
    """
    path = os.path.join(_OUT_DIR, f"results_whittle_sync_seed{seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)


def plot_average_rewards(avg_qwi, avg_random, avg_oracle, gamma, epsilon, timesteps, SEED,
                        tail_frac=0.05, min_tail=1000, heterogeneous=False):
    """
    Plot average rewards for QWI, Random, and Oracle policies.
    
    Includes:
    - Main plot of full trajectory
    - Zoomed inset of final portion
    
    Parameters
    ----------
    avg_qwi : list
        Average rewards for QWI policy
    avg_random : list
        Average rewards for random policy
    avg_oracle : list
        Average rewards for oracle policy
    gamma : float
        Discount factor
    epsilon : float
        Exploration rate
    timesteps : int
        Total timesteps
    SEED : int
        Random seed
    tail_frac : float
        Fraction of trajectory to show in zoom
    min_tail : int
        Minimum timesteps for zoom window
    heterogeneous : bool
        Whether dynamics are heterogeneous
    """
    n = len(avg_qwi)
    tail = int(max(min_tail, tail_frac * timesteps))
    tail = min(tail, n)
    start = n - tail
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    # Main curves
    ax.plot(t, avg_qwi, label='QWI', linewidth=2, color="blue", linestyle="-")
    ax.plot(t, avg_random, label='Random', linewidth=2, color="red", linestyle="-")
    ax.plot(t, avg_oracle, label='Oracle (Whittle)', linewidth=2, color="black", linestyle="--")

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoomed inset
    zoom_ax = inset_axes(ax, width="42%", height="42%", loc="center", borderpad=1.2)
    zoom_ax.plot(t[start:], avg_qwi[start:], linewidth=1.8, label='QWI', color="blue", linestyle="-")
    zoom_ax.plot(t[start:], avg_random[start:], linewidth=1.8, label='Random', color="red", linestyle="-")
    zoom_ax.plot(t[start:], avg_oracle[start:], linewidth=1.8, label='Oracle', color="black", linestyle="--")
    zoom_ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    zoom_ax.grid(True, alpha=0.3)
    
    # Tight y-limits on zoom
    ymin = min(np.min(avg_qwi[start:]), np.min(avg_random[start:]), np.min(avg_oracle[start:]))
    ymax = max(np.max(avg_qwi[start:]), np.max(avg_random[start:]), np.max(avg_oracle[start:]))
    pad = 0.05 * (ymax - ymin + 1e-12)
    zoom_ax.set_ylim(ymin - pad, ymax + pad)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.12)

    fname = f'avg_reward_sd{SEED}_hetro:{heterogeneous}_itr{timesteps}_gamma{gamma}_eps{epsilon}.png'
    _savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_results_index(avg_qwi, avg_random, cumulative_qwi, cumulative_random,
                      indices_history, optimal_indices, nStates, gamma, epsilon, 
                      timesteps, SEED, heterogeneous=False):
    """
    Plot Whittle index convergence.
    
    For homogeneous case: plots all states for the single arm.
    For heterogeneous case: plots only the first arm.
    
    Parameters
    ----------
    indices_history : dict or dict of dict
        For homogeneous: {state: [values]}
        For heterogeneous: {user_id: {state: [values]}}
    optimal_indices : np.ndarray or dict
        For homogeneous: array of optimal indices
        For heterogeneous: {user_id: array}
    """
    colors = plt.cm.tab10(np.linspace(0, 1, nStates))

    if not heterogeneous:
        # Single arm plot
        plt.figure(figsize=(10, 5))
        for s in range(nStates):
            plt.plot(indices_history[s], label=f'State {s+1} (Learned)',
                    color=colors[s], linewidth=2, alpha=0.7)
            plt.axhline(y=optimal_indices[s], color=colors[s], linestyle='--',
                       linewidth=2, label=f'State {s+1} (Optimal)')

        plt.xlabel('Timestep')
        plt.ylabel('Whittle Index')
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fname = f'whittle_index_convergence_sync_sd{SEED}_heteroFalse_itr{timesteps}_gamma{gamma}_eps{epsilon}.png'
        _savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()
        return

    # Heterogeneous: plot first arm only
    first_user_id = min(indices_history.keys())
    per_state_hist = indices_history[first_user_id]

    plt.figure(figsize=(10, 5))
    for s in range(nStates):
        plt.plot(per_state_hist[s], label=f'State {s+1} (Learned)',
                color=colors[s], linewidth=2, alpha=0.7)
        opt = optimal_indices[first_user_id][s]
        plt.axhline(y=opt, color=colors[s], linestyle='--',
                   linewidth=2, label=f'State {s+1} (Optimal)')

    plt.xlabel('Timestep')
    plt.ylabel('Whittle Index')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname = f'whittle_index_convergence_sync_sd{SEED}_heteroTrue_arm{first_user_id}_itr{timesteps}_gamma{gamma}_eps{epsilon}.png'
    _savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()


def _stack(curve_list):
    """Stack a list of 1D curves into [n_seeds, T]."""
    return np.vstack([np.asarray(c, dtype=float) for c in curve_list])


def plot_aggregate_avg_reward(avg_qwi_list, avg_rand_list, avg_oracle_list,
                              timesteps, gamma, epsilon, heterogeneous, seeds,
                              tail_frac=0.05, min_tail=1000):
    """
    Plot mean ± std of average reward curves across seeds.
    
    Parameters
    ----------
    avg_qwi_list : list of arrays
        QWI rewards for each seed
    avg_rand_list : list of arrays
        Random rewards for each seed
    avg_oracle_list : list of arrays
        Oracle rewards for each seed
    timesteps : int
        Number of timesteps
    gamma : float
        Discount factor
    epsilon : float
        Exploration rate
    heterogeneous : bool
        Heterogeneous dynamics flag
    seeds : list
        List of seeds used
    """
    A_q = _stack(avg_qwi_list)
    A_r = _stack(avg_rand_list)
    A_o = _stack(avg_oracle_list)

    m_q, s_q = A_q.mean(axis=0), A_q.std(axis=0)
    m_r, s_r = A_r.mean(axis=0), A_r.std(axis=0)
    m_o, s_o = A_o.mean(axis=0), A_o.std(axis=0)

    T = len(m_q)
    t = np.arange(T)

    tail = int(max(min_tail, tail_frac * timesteps))
    tail = min(tail, T)
    start = T - tail

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    
    # QWI
    ax.plot(t, m_q, label='QWI (mean)', linewidth=2, color="blue")
    ax.fill_between(t, m_q - s_q, m_q + s_q, alpha=0.2)
    
    # Random
    ax.plot(t, m_r, label='Random (mean)', linewidth=2, color="red")
    ax.fill_between(t, m_r - s_r, m_r + s_r, alpha=0.2)
    
    # Oracle
    ax.plot(t, m_o, label='Oracle (mean)', linewidth=2, color="black", linestyle="--")
    ax.fill_between(t, m_o - s_o, m_o + s_o, alpha=0.2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Zoomed inset
    zoom_ax = inset_axes(ax, width="42%", height="42%", loc="center", borderpad=1.2)
    zoom_ax.plot(t[start:], m_q[start:], linewidth=1.8, color="blue")
    zoom_ax.fill_between(t[start:], m_q[start:] - s_q[start:], m_q[start:] + s_q[start:], alpha=0.2)
    zoom_ax.plot(t[start:], m_r[start:], linewidth=1.8, color="red")
    zoom_ax.fill_between(t[start:], m_r[start:] - s_r[start:], m_r[start:] + s_r[start:], alpha=0.2)
    zoom_ax.plot(t[start:], m_o[start:], linewidth=1.8, color="black", linestyle="--")
    zoom_ax.fill_between(t[start:], m_o[start:] - s_o[start:], m_o[start:] + s_o[start:], alpha=0.2)
    zoom_ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    zoom_ax.grid(True, alpha=0.3)
    
    ymin = min(np.min(m_q[start:]), np.min(m_r[start:]), np.min(m_o[start:]))
    ymax = max(np.max(m_q[start:]), np.max(m_r[start:]), np.max(m_o[start:]))
    pad = 0.05 * (ymax - ymin + 1e-12)
    zoom_ax.set_ylim(ymin - pad, ymax + pad)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.12)

    tag = f"{len(seeds)}seeds_{min(seeds)}-{max(seeds)}"
    fname = f"avg_reward_AGG_{tag}_hetro:{heterogeneous}_itr{timesteps}_gamma{gamma}_eps{epsilon}.png"
    _savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_aggregate_indices(results_all, nStates, timesteps, gamma, epsilon, 
                          heterogeneous, seeds):
    """
    Plot aggregated Whittle index convergence across seeds.
    
    Shows mean ± std of learned indices compared to optimal values.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, nStates))
    T = len(results_all[0]["avg_qwi"])
    t = np.arange(T)

    plt.figure(figsize=(10, 5))

    if not heterogeneous:
        for s in range(nStates):
            A = np.vstack([np.asarray(r["indices_history"][s], dtype=float) 
                          for r in results_all])
            m, sd = A.mean(axis=0), A.std(axis=0)

            plt.plot(t, m, label=f'State {s+1} (mean)', color=colors[s], 
                    linewidth=2, alpha=0.9)
            plt.fill_between(t, m - sd, m + sd, color=colors[s], alpha=0.2)

            opt = results_all[0]["optimal_indices"][s]
            plt.axhline(opt, color=colors[s], linestyle='--', linewidth=1.8,
                       label=f'State {s+1} (Optimal)')
    else:
        # Find common user across all seeds
        common_users = set(results_all[0]["indices_history"].keys())
        for r in results_all[1:]:
            common_users &= set(r["indices_history"].keys())
        
        first_user_id = min(common_users)

        for s in range(nStates):
            A = np.vstack([
                np.asarray(r["indices_history"][first_user_id][s], dtype=float)
                for r in results_all
            ])
            m, sd = A.mean(axis=0), A.std(axis=0)

            plt.plot(t, m, label=f'State {s+1} (mean)', color=colors[s], 
                    linewidth=2, alpha=0.9)
            plt.fill_between(t, m - sd, m + sd, color=colors[s], alpha=0.2)

            opts = np.array([r["optimal_indices"][first_user_id][s] 
                           for r in results_all], dtype=float)
            opt_mean = opts.mean()

            plt.axhline(opt_mean, color=colors[s], linestyle='--', linewidth=1.8,
                       label=f'State {s+1} (Optimal mean)')

    plt.xlabel('Timestep')
    plt.ylabel('Whittle Index')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    tag = f"{len(seeds)}seeds_{min(seeds)}-{max(seeds)}"
    het_tag = "heteroTrue" if heterogeneous else "heteroFalse"
    fname = f'whittle_index_AGG_{het_tag}_{tag}_itr{timesteps}_gamma{gamma}_eps{epsilon}.png'
    _savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()