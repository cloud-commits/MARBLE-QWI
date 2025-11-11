import numpy as np

class QWI_MARBLE:
    def __init__(self, nUsers, gamma, env, stateTable, nActions, P_true=None, R_true=None):
        self.env = env
        self.stateTable = stateTable
        self.nStates = np.shape(stateTable)[0]
        self.nUsers = nUsers
        self.nActions = nActions
        self.P_true = P_true    # [S, A, S] or None
        self.R_true = R_true    # [S, A]     or None
        
        self.indices = np.zeros(self.nStates)
        
        self.qTable = {
            z: np.zeros((self.nStates, self.nActions)) 
            for z in range(self.nStates)
        }
        
        self.indexCounter = 0
        self.gamma = gamma
        
        # Store transition counts for learning rates
        self.stateActionCounters = {
            z: np.zeros((self.nStates, self.nActions), dtype=int)
            for z in range(self.nStates)
        }
        
        # Store environment dynamics (empirical estimates)
        self.transition_counts = np.zeros((self.nStates, self.nActions, self.nStates))
        self.reward_sums = np.zeros((self.nStates, self.nActions))
        self.state_action_visits = np.zeros((self.nStates, self.nActions))

    def get_state_index(self, state):
        stateLocation = np.where((self.stateTable == state))[0][0]
        return stateLocation

    def get_an(self, stateActionCounter, constant1=0.1):
        stepSize = constant1 / (np.ceil((stateActionCounter + 1) / 10000))

        return stepSize
    
    def get_bn(self, constant2=0.001):
        n = self.indexCounter + 1
        stepSize = constant2 / (1 + np.ceil((n * np.log1p(n)) / 10000))

        return stepSize

    def collect_experience(self, currentState, nextState, action, reward):
        current_idx = self.get_state_index(currentState)
        next_idx = self.get_state_index(nextState)
        
        self.transition_counts[current_idx, action, next_idx] += 1
        self.reward_sums[current_idx, action] += reward
        self.state_action_visits[current_idx, action] += 1
        
        for z in range(self.nStates):
            self.stateActionCounters[z][current_idx, action] += 1

    def synchronous_qTable_update(self):
        
        # If we were given the true averaged model, use it
        if self.P_true is not None and self.R_true is not None:
            P_hat = self.P_true
            # R_hat = self.R_true
        else:
            # Fallback: build empirical estimates (your current code)
            P_hat = np.zeros((self.nStates, self.nActions, self.nStates))
            # R_hat = np.zeros((self.nStates, self.nActions))
            for s in range(self.nStates):
                for a in range(self.nActions):
                    if self.state_action_visits[s, a] > 0:
                        P_hat[s, a, :] = self.transition_counts[s, a, :] / self.state_action_visits[s, a]
                        # R_hat[s, a] = self.reward_sums[s, a] / self.state_action_visits[s, a]
                    else:
                        P_hat[s, a, :] = 1.0 / self.nStates
                        # R_hat[s, a] = 0.0
        
        # Use the current environment e to build R_hat(s,a) = reward(s,a,e)
        e = self.env.get_current_env()
        rm = self.env.reward_matrices[e]
        R_hat = np.zeros((self.nStates, self.nActions))
        for s in range(self.nStates):
            for a in range(self.nActions):
                R_hat[s, a] = rm[(s, a)]  # rewards are 0-based in your dicts

        old_qTable = {z: self.qTable[z].copy() for z in range(self.nStates)}
        
        for z in range(self.nStates):
            for s in range(self.nStates):
                for a in range(self.nActions):
                    _an = self.get_an(self.stateActionCounters[z][s, a])
                    
                    s_next = np.random.choice(self.nStates, p=P_hat[s, a, :])

                    expected_future_value = np.max(old_qTable[z][s_next, :])

                    td_target = (R_hat[s, a] + 
                                self.indices[z] * (1 - a) + 
                                self.gamma * expected_future_value)
                    
                    self.qTable[z][s, a] += _an * (td_target - old_qTable[z][s, a])

    def update_index(self, state):
        current_idx = self.get_state_index(state)
        b_n = self.get_bn()
        
        for z in range(self.nStates):
            diff = self.qTable[z][z, 1] - self.qTable[z][z, 0]
            self.indices[z] += b_n * diff
        
        self.indexCounter += 1


def create_state_table(nStates):
    return np.arange(1, nStates + 1, dtype=np.uint32)


