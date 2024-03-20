import numpy as np
import copy

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
        self.R = copy.deepcopy(R)
        self.Q = np.zeros(np.array(R).shape)
        self.prime_state = 0
        self.current_state = None
        self.prev_state = None

        self.position = [None, None]
        self.rewards = []
        self.total_reward = 0

    def epsilon_greedy(self, valid_actions):
        
        self.valid_a = valid_actions
        # print(valid_actions)
        q_values = [self.Q[self.prime_state][self.current_state, action] for action in valid_actions]
        #print(f'q_vals: {q_values}')
        best_actions = valid_actions[np.where(q_values == np.max(q_values))[0]]
        best_actions_q_values = [self.Q[self.prime_state][self.current_state, best_action] for best_action in best_actions]

        if np.random.uniform() > self.epsilon:
            self.action_taken = np.random.choice(valid_actions)
        else:
            self.action_taken = np.random.choice(best_actions)
            
        #print(f'action taken {self.action_taken}')

        reward = self.R[self.prime_state][self.current_state, self.action_taken]
        self.prev_state = self.current_state
        self.current_state = self.action_taken

        self.rewards.append(reward)
        self.total_reward += reward

        #print(f'from {self.prev_state} to {self.current_state}')

    def epsilon_test(self, valid_actions):
        self.valid_a = valid_actions
        q_values = [self.Q[self.prime_state][self.current_state, action] for action in valid_actions]
        best_actions = valid_actions[np.where(q_values == np.max(q_values))[0]]

        #print(f'Best actions: {best_actions}')

        self.action_taken = np.random.choice(best_actions)
        
        reward = self.R[self.prime_state][self.current_state, self.action_taken]
        self.prev_state = self.current_state
        self.current_state = self.action_taken

        self.total_reward += reward

    def update_q(self):
        #if self.prev_state == 64:
        #    print(f'updates:\n - prevstate: {self.prev_state}\n - actiontaken: {self.action_taken}\n -validactions: {self.valid_a}')

        self.Q[self.prime_state][self.prev_state, self.action_taken] = self.Q[self.prime_state][self.prev_state, self.action_taken] + (self.alpha * (self.R[self.prime_state][self.prev_state, self.action_taken] + (self.gamma * np.max(self.Q[self.prime_state][self.current_state])) - self.Q[self.prime_state][self.prev_state, self.action_taken]))
        # if self.prev_state == 64 or self.prev_state == 65 or self.prev_state == 66:
        #     print(f'{self.Q[self.prev_state, self.action_taken]}')

    def set_state(self, state):
        self.current_state = state

    def set_position(self, pos):
        self.position = pos

    def update_env(self, R):
        self.R = copy.deepcopy(R)

    def set_prime_state(self, prime_state):

        self.prime_state = prime_state