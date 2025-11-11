import numpy as np

class MARBLEEnv:

    def __init__(self, seed, gamma, nStates, nActions, transition_matrices, reward_matrices, H):

        self.rng = np.random.default_rng(seed)
        self.nStates = nStates
        self.nActions = nActions
        self.transition_matrices = transition_matrices
        self.reward_matrices = reward_matrices
        self.H = H
        self.env_states = list(H.keys())
        self.env_index = {e: i for i, e in enumerate(self.env_states)}
        self.env_state = self.rng.choice(self.env_states)
        self.gamma = gamma
        self.userState = int(self.rng.integers(1, self.nStates + 1))


    def reset(self):
        self.userState = int(self.rng.integers(1, self.nStates + 1))
        self.env_state  = self.rng.choice(self.env_states)
        return np.array([self.userState])


    def update_environment(self):
        current_env = self.env_state
        probs = list(self.H[current_env].values())
        self.env_state = self.rng.choice(self.env_states, p=probs)


    def step(self, action):

        s = self.userState
        env = self.env_state
        T = self.transition_matrices[env][action]

        next_state = int(self.rng.choice(self.nStates, p=T[s - 1])) + 1

        reward = self.reward_matrices[env][(s - 1, action)]
    
        self.userState = next_state
        self.update_environment()

        return np.array([next_state]), reward
    
    def get_current_env(self):
        return self.env_state

def define_environment_transition(): 
    return { 
        'E1': {'E1': 0.85, 'E2': 0.15},  # Good env
        'E2': {'E1': 0.15, 'E2': 0.85}   # Bad env
    }


def define_transition_matrices(): 
    return { 
        'E1': {  # Good environment
            0: np.array([  # Action 0: No recommendation
                [0.7, 0.2, 0.08, 0.02],  
                [0.3, 0.4, 0.25, 0.05],  
                [0.2, 0.3, 0.35, 0.15], 
                [0.15, 0.25, 0.3, 0.3]   
            ]), 
            1: np.array([  # Action 1: Send recommendation
                [0.5, 0.35, 0.12, 0.03],
                [0.15, 0.3, 0.4, 0.15],  
                [0.1, 0.2, 0.4, 0.3],   
                [0.1, 0.2, 0.3, 0.4]  
            ]) 
        }, 
        'E2': {  # Bad environment
            0: np.array([  # Action 0: No recommendation
                [0.85, 0.12, 0.025, 0.005], 
                [0.5, 0.35, 0.12, 0.03],  
                [0.4, 0.35, 0.2, 0.05],    
                [0.4, 0.3, 0.2, 0.1]    
            ]), 
            1: np.array([  # Action 1: Send recommendation
                [0.7, 0.2, 0.08, 0.02],    
                [0.35, 0.35, 0.22, 0.08],
                [0.25, 0.3, 0.3, 0.15],   
                [0.3, 0.3, 0.25, 0.15]  
            ]) 
        } 
    }



def define_reward_matrices():
    reward_matrices = {
        'E1': {  # Good environment (users more receptive)
            (0, 0): 0.0,   
            (0, 1): 0.05, 
            (1, 0): 0.1,  
            (1, 1): 0.25,  
            (2, 0): 0.2,   
            (2, 1): 0.4,  
            (3, 0): 0.99,  
            (3, 1): 0.8   
        },
        'E2': {  # Bad environment (users less receptive, more recommendation fatigue)
            (0, 0): 0.0,   
            (0, 1): 0.05,
            (1, 0): 0.08,  
            (1, 1): 0.15,  
            (2, 0): 0.15,  
            (2, 1): 0.25,  
            (3, 0): 0.99,   
            (3, 1): 0.6   
            }
    }
    # Convert costs -> rewards: best ~ 0, worst more negative
    for env in reward_matrices:
        for key in reward_matrices[env]:
            reward_matrices[env][key] = -reward_matrices[env][key]
    return reward_matrices


def make_user_dynamics(user_id, base_transitions, base_rewards, *, seed, heterogeneous=False,
                       trans_conc=400.0, reward_sigma=0.10):
    """
    Return (transitions_i, rewards_i) for a given user.
    If heterogeneous=False: returns the base dicts (no change).
    If heterogeneous=True:
      - Transitions: small Dirichlet row-perturbations around each base row.
      - Rewards: small lognormal multiplicative noise, then clipped to [0, 1].
    H (the environment MC over E1/E2) stays the same for all users.
    """
    if not heterogeneous:
        return base_transitions, base_rewards

    rng = np.random.default_rng(seed + 12345 + user_id)

    # Transitions: per-env, per-action, per-row Dirichlet centered at base row
    transitions_i = {}
    for e, acts in base_transitions.items():
        transitions_i[e] = {}
        for a, T in acts.items():  # T: [S,S]
            S = T.shape[0]
            T_pert = np.zeros_like(T)
            for s in range(S):
                row = T[s]
                alpha = np.maximum(row * trans_conc, 1e-8)  # concentration controls variance
                T_pert[s] = rng.dirichlet(alpha)
            transitions_i[e][a] = T_pert

    # Rewards: multiplicative lognormal noise
    rewards_i = {}
    for e, Rdict in base_rewards.items():
        new_R = {}
        for (s, a), r in Rdict.items():
            mult = rng.lognormal(mean=0.0, sigma=reward_sigma)  # around 1.0
            r_new = float(np.clip(r * mult, -1.0, 0.0))
            new_R[(s, a)] = r_new
        rewards_i[e] = new_R

    return transitions_i, rewards_i

