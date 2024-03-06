import numpy as np

# posible actions
# keep track of ores

# diamond_ores_collected = [false, false, false]
# if false in diamond_ores_collected:
    # exit not open

# np.where(dimaond_ores_collected == false)
# if mine dont thing


class Agent():
    def __init__(self, alpha, gamma, epsilon, R):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.R = R
        self.Q = np.zeros(R.shape)
        self.current_state = None
        self.prev_state = None

    def take_action(self, valid_actions):
        # print(valid_actions)
        q_values = [self.Q[self.current_state, action] for action in valid_actions]
        #print(f'q_vals: {q_values}')
        best_actions = valid_actions[np.where(q_values == np.max(q_values))[0]]
        best_actions_q_values = [self.Q[self.current_state, best_action] for best_action in best_actions]

        if np.random.uniform() > self.epsilon:
            self.action_taken = np.random.choice(valid_actions)
        else:
            self.action_taken = np.random.choice(best_actions)
            
        #print(f'action taken {self.action_taken}')

        reward = self.R[self.current_state, self.action_taken]
        self.prev_state = self.current_state
        self.current_state = self.action_taken

        #print(f'from {self.prev_state} to {self.current_state}')

    def update_q(self):
        self.Q[self.prev_state, self.action_taken] = self.Q[self.prev_state, self.action_taken] + (self.alpha * self.R[self.prev_state, self.action_taken] + (self.gamma * np.max(self.Q[self.current_state])) - self.Q[self.prev_state, self.action_taken])

    def set_position(self, state):
        self.current_state = state