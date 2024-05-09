# ================================================================================= Imports

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import os

# ================================================================================= Setup

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Device: {device}')

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# ================================================================================= Agent

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_Agent():
    def __init__(self, agent_name, n_observations, n_actions, double_dqn, memory_capacity, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr):
        self.name = agent_name
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(memory_capacity, self.Transition)
        self.online_network = DQN(n_observations, n_actions).to(device)
        self.target_network = DQN(n_observations, n_actions).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.double_dqn = double_dqn

    def __getstate__(self):
        return (self.name, self.online_network, self.target_network, self.double_dqn)

    def __setstate__(self, state):
        self.name, self.online_network, self.target_network, self.double_dqn = state

    def greedy_action(self, state):
        return self.online_network(state).max(1).indices.view(1, 1)

    def explore_action(self, action_space):
        return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)

    def evaluate_actions(self, states, actions, rewards, dones, gamma):
        if self.double_dqn:
            next_q_values = self.target_network(states).gather(dim=1, index=actions)
        else:
            next_q_values = self.online_network(states).gather(dim=1, index=actions)
            
        q_values = rewards + (gamma * next_q_values * (1 - dones))
        return q_values
        
    def update_q_values(self, states, rewards, dones, gamma):
        actions = self.greedy_action(states)
        q_values = self.evaluate_actions(states, actions, rewards, dones, gamma)
        return q_values



class Trainer():
    def __init__(self, agent, env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, optimizer, plot_results=True, seed=False, save_agent=True):

        self.agent = agent
        self.env = env
        self.save_agent_bool = save_agent

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.epsilon = 0
        self.TAU = tau
        self.LR = lr
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        self.optimizer = optimizer

        self.steps_done = 0

        self.episode_durations = []

    ## Remember to update steps after each action

    def epsilon_policy(self, state):
        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if self._random_state.random() < self.epsilon:
            action = self.agent.explore_action(self.env.action_space)
        else:
            action = self.agent.greedy_action(state)
        return action

    def optimize_models(self):
        if len(self.agent.memory) < self.BATCH_SIZE:
            return
        transitions = self.agent.memory.sample(self.BATCH_SIZE)
        batch = self.agent.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.agent.online_network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.agent.target_network(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.agent.online_network.parameters(), 100)
        self.optimizer.step()

    def plot_steps(self, training=True):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if not training:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            if training:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def save_agent(self, directory='checkpoints'):
        print('Saving agent...')
        dir = os.path.join(os.getcwd(), directory, self.agent.name)
        print(f'Saving agent to: {dir}')
        with open(f'{dir}.pkl', 'wb') as file:
            pickle.dump(self.agent, file)

    def train(self, num_episodes):
        plt.ion()
        for i_episode in range(num_episodes):
            print(f'Episode: {i_episode}/{num_episodes}')
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.epsilon_policy(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.agent.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_models()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.agent.target_network.state_dict()
                online_net_state_dict = self.agent.online_network.state_dict()
                for key in online_net_state_dict:
                    target_net_state_dict[key] = online_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.agent.target_network.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_steps()
                    break
        if self.save_agent_bool:
            self.save_agent()

        print('Training complete')
        self.plot_steps(training=False)
        plt.ioff()
        plt.show()
        

class Tester():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def test(self):
        self.agent.online_network.eval()
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = self.agent.greedy_action(state).item()
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if done:
                print(f'agent took {t} step to land')
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

# ================================================================================= Training Function

def load_agent(agent_dir):
    print('Loading agent...')
    dir = os.path.join(os.getcwd(), agent_dir)
    print(f'Loading agent from: {dir}')
    with open(f'{dir}.pkl', 'rb') as file:
        agent = pickle.load(file)

    return agent


# ================================================================================= Training Section

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

REPLAY_MEMORY = 10000


env = env = gym.make("LunarLander-v2",
    continuous = False,
    gravity= -10.0,
    enable_wind= False,
    wind_power = 0.0,
    turbulence_power = 1.5,
    render_mode= "human"
)

state, info = env.reset()

n_observations = len(state)
n_actions = env.action_space.n

# agent = DQN_Agent("double_dqn_agent_1kep", n_observations, n_actions, True, 10000, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

agent = load_agent(os.path.join('checkpoints', 'double_dqn_agent_1kep'))

optimizer = optim.AdamW(agent.online_network.parameters(), lr=LR, amsgrad=True)

# trainer = Trainer(agent, env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, optimizer, plot_results=True, seed=False)
# trainer.train(num_episodes = 1000)

tester = Tester(agent, env)
tester.test()