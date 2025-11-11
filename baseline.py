import numpy as np

class TheoreticalBaseline:

    def __init__(self, nStates, gamma, nActions, transitions, rewards, H, ):
        self.nStates = nStates
        self.nActions = nActions
        self.transitions = transitions
        self.rewards = rewards
        self.H = H
        self.gamma = gamma
        self.env_states = list(H.keys())
        
        self.mu_E = self.compute_stationary_distribution()
        
        self.avg_transitions, self.avg_rewards = self.compute_averaged_dynamics()
        
        self.optimal_indices = None
        self.optimal_Q = None
        
        self.env_specific_indices = None
        
    def compute_stationary_distribution(self):
 
        n_env = len(self.env_states)
        
        P = np.zeros((n_env, n_env))
        for i, e in enumerate(self.env_states):
            for j, e_next in enumerate(self.env_states):
                P[i, j] = self.H[e][e_next]
        
        A = P.T - np.eye(n_env)
        A = np.vstack([A, np.ones(n_env)])
        b = np.zeros(n_env + 1)
        b[-1] = 1
        
        mu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        mu_E = {e: mu[i] for i, e in enumerate(self.env_states)}
        
        
        return mu_E
    
    def compute_averaged_dynamics(self):
        avg_transitions = {}
        avg_rewards = {}
        
        for a in range(self.nActions):
            avg_T = np.zeros((self.nStates, self.nStates))
            for e in self.env_states:
                avg_T += self.mu_E[e] * self.transitions[e][a]
            avg_transitions[a] = avg_T
        
        for s in range(self.nStates):
            for a in range(self.nActions):
                avg_r = sum(self.mu_E[e] * self.rewards[e][(s, a)] 
                           for e in self.env_states)
                avg_rewards[(s, a)] = avg_r
        
        return avg_transitions, avg_rewards
    
    # def compute_optimal_whittle_indices(self, max_iter=10000, tol=1e-6):
    #     optimal_indices = np.zeros(self.nStates)
    #     optimal_Q = {z: np.zeros((self.nStates, self.nActions)) 
    #                  for z in range(self.nStates)}
        
        
    #     for z in range(self.nStates):
    #         lambda_min, lambda_max = -10.0, 10.0
            
    #         for _ in range(50):
    #             lambda_mid = (lambda_min + lambda_max) / 2
    #             Q = self.value_iteration_averaged(z, lambda_mid, max_iter, tol)
    #             gap = Q[z, 1] - Q[z, 0]
                
    #             if abs(gap) < tol:
    #                 break
    #             elif gap > 0:
    #                 lambda_min = lambda_mid
    #             else:
    #                 lambda_max = lambda_mid
            
    #         optimal_indices[z] = lambda_mid
    #         optimal_Q[z] = Q
        
    #     self.optimal_indices = optimal_indices
    #     self.optimal_Q = optimal_Q
        
    #     return optimal_indices, optimal_Q
    


    def compute_optimal_whittle_indices(self, max_iter=10_000, tol=1e-6):
        """
        Compute the (oracle) Whittle index λ*(z) for each state z by solving:
            f(λ; z) = Q_λ(z, 1) - Q_λ(z, 0) = 0
        on the environment-averaged single-arm MDP.
        We assume MAI so f(λ; z) is (non)increasing and has a unique zero.
        """

        # 1) Outputs we’ll fill:
        #    - optimal_indices[z]  : the λ*(z) we find by bisection
        #    - optimal_Q[z]        : the Q-table at λ*(z) (useful for diagnostics)
        optimal_indices = np.zeros(self.nStates)
        optimal_Q = {z: np.zeros((self.nStates, self.nActions)) for z in range(self.nStates)}

        # 2) Helper that, given λ, returns the gap f(λ; z) and the Q used to compute it.
        #    (Q is obtained by value iteration under the *averaged* model.)
        def gap_at_lambda(lambda_val, z):
            Q = self.value_iteration_averaged(lambda_val, max_iter=max_iter, tol=tol)
            gap = Q[z, 1] - Q[z, 0]   # f(λ; z) = Q(z, active) - Q(z, passive)
            return gap, Q

        # 3) Loop over reference states z = 0..nStates-1
        for z in range(self.nStates):

            # ---- 3a) Robust bracket: find [λ_min, λ_max] so that f(λ_min) and f(λ_max) have opposite signs ----
            # Start with a symmetric bracket [-width, +width]. If no sign change, expand (double) 'width'.
            width = 10.0
            for _ in range(20):  # expand up to 20 times if needed
                lam_min, lam_max = -width, width
                g_min, _ = gap_at_lambda(lam_min, z)   # f(λ_min; z)
                g_max, _ = gap_at_lambda(lam_max, z)   # f(λ_max; z)
                # A root is bracketed if the function changes sign over the interval
                if g_min <= 0 <= g_max or g_max <= 0 <= g_min:
                    break
                width *= 2.0
            else:
                # If we never broke out, we failed to bracket a root.
                raise RuntimeError(f"Could not bracket Whittle index root for state z={z}")

            # ---- 3b) Bisection over the bracket until f(λ; z) ~ 0 (or bracket is tiny) ----
            Q_mid = None  # we’ll keep the last Q(·; λ_mid) we computed
            for _ in range(10_000):  # plenty for double precision with a ~20-bit bracket
                lam_mid = 0.5 * (lam_min + lam_max)  # midpoint
                g_mid, Q_mid = gap_at_lambda(lam_mid, z)

                # Stop if the gap is small enough OR the interval is tiny
                if abs(g_mid) < tol or (lam_max - lam_min) < 1e-8:
                    break

                # MAI implies f is monotone in λ. If f(λ_mid) > 0, move left bound up; else move right bound down.
                if g_mid > 0:
                    lam_min = lam_mid
                else:
                    lam_max = lam_mid

            # ---- 3c) Save results for this z ----
            optimal_indices[z] = lam_mid  # approximate root λ*(z)
            optimal_Q[z] = Q_mid          # Q evaluated at that λ

        # 4) Cache on the object and return
        self.optimal_indices = optimal_indices
        self.optimal_Q = optimal_Q
        return optimal_indices, optimal_Q


    def compute_environment_specific_indices(self, max_iter=10000, tol=1e-6):

        env_specific_indices = {}
                
        for env in self.env_states:
            indices_env = np.zeros(self.nStates)
            
            for z in range(self.nStates):
                lambda_min, lambda_max = -10.0, 10.0
                
                for _ in range(50):
                    lambda_mid = (lambda_min + lambda_max) / 2
                    Q = self.value_iteration_env_specific(z, lambda_mid, env, max_iter, tol)
                    gap = Q[z, 1] - Q[z, 0]
                    
                    if abs(gap) < tol:
                        break
                    elif gap > 0:
                        lambda_min = lambda_mid
                    else:
                        lambda_max = lambda_mid
                
                indices_env[z] = lambda_mid
            
            env_specific_indices[env] = indices_env
        
        self.env_specific_indices = env_specific_indices
        return env_specific_indices
    
    def value_iteration_averaged(self, lambda_val, max_iter=10000, tol=1e-6):
        Q = np.zeros((self.nStates, self.nActions))
        
        for iteration in range(max_iter):
            Q_old = Q.copy()
            
            for s in range(self.nStates):
                for a in range(self.nActions):
                    immediate = self.avg_rewards[(s, a)] + lambda_val * (1 - a)
                    future = 0.0
                    for s_next in range(self.nStates):
                        p = self.avg_transitions[a][s, s_next]
                        future += p * np.max(Q_old[s_next, :])
                    Q[s, a] = immediate + self.gamma * future
            
            if np.max(np.abs(Q - Q_old)) < tol:
                break
        
        return Q
    
    def value_iteration_env_specific(self, z, lambda_val, env, max_iter=10000, tol=1e-6):

        Q = np.zeros((self.nStates, self.nActions))
        
        for iteration in range(max_iter):
            Q_old = Q.copy()
            
            for s in range(self.nStates):
                for a in range(self.nActions):
                    immediate = self.rewards[env][(s, a)] + lambda_val * (1 - a)
                    future = 0.0
                    for s_next in range(self.nStates):
                        p = self.transitions[env][a][s, s_next]
                        future += p * np.max(Q_old[s_next, :])
                    Q[s, a] = immediate + self.gamma * future
            
            if np.max(np.abs(Q - Q_old)) < tol:
                break
        
        return Q

