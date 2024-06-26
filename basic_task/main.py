import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import pickle
import os
import json
from collections import deque

from environment import Environment
from agent import Agent

grid_x = 8  # Grid size
grid_y = 8
diamond_collected_reward = 50
corrosive_fume_reward = -30
exit_reward = 100
start_exit_pos = [[7, 3]]
wall_rail_pos = [[0, 1], [1, 1], [2, 1], [4, 1], [5, 1], [2, 2], [1, 3], [2, 3], [4, 4], [5, 4], [6, 4], [2, 6], [4, 6], [5, 6], [6, 6], [5, 0], [6, 0], [1, 7]]
diamond_ore_pos = [[0, 0], [0, 5], [5, 5]]
corrosive_fumes_pos = [[1, 2], [6, 5], [0, 7]]
rails_mapping_pos = [[[6, 1], [3, 0]], [[4, 0], [6, 2]], [[2, 7], [1, 5]], [[1, 6], [3, 7]]]

env = Environment(grid_x, grid_y, diamond_collected_reward, corrosive_fume_reward, exit_reward, start_exit_pos, wall_rail_pos, diamond_ore_pos, corrosive_fumes_pos, rails_mapping_pos)
env.generate_R_matrix()

########################## Visualisation
# Black = Agent                       1000
# Brown = Walls                       -5
# Blue = Diamonds                     50
# Purple = Fumes                      -30
# Red = Rails                         5
# White = Empty Tiles                 0
# Yellow = Terminal                   100
# Orange = Mining                     1100
# Green = Success                     1200

def visualisePath(images, agent_rewards, name):
    
    print(f'Generating Path... {len(images)} frames')

    fig = plt.figure()
    frames = []

    bounds = [-31, -6, -1, 1, 6, 51, 101, 1001, 1101, 1201]
    cmap = mpl.colors.ListedColormap(['purple','brown','white', 'red', 'blue', 'yellow', 'black', 'orange', 'green'])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for idx, img in enumerate(images):
        
        frames.append([plt.imshow(img, interpolation='nearest', cmap = cmap,norm=norm, animated = True), plt.annotate(f'Step: {idx}, R: {agent_rewards[idx - 1]}', (0, 0))])

    
    ani = animation.ArtistAnimation(fig, frames)
    ani.save(f'./agents/{name}/path.mp4')

def plot_reward(rewards, name):

    img = plt.plot(rewards)

    plt.savefig(f'./agents/{name}/rewards.png')


def train_agent(alpha, gamma, epsilon, episodes, timesteps):
    name = f'a{alpha}g{gamma}e{epsilon}'.replace('.', '')
    decay = 0.99999 #epsilon/timesteps
    
    if not os.path.exists(f'./agents/{name}'):
        os.makedirs(f'./agents/{name}')

    agent = Agent(alpha, gamma, epsilon, decay, env.R, name)

    ep_rewards = []
    ep_steps = []

    for episode in range(episodes):
        print(f'{name} - Starting Episode {episode}:')
        env.reset_env()

        agent.set_state(env.start_state)
        agent.total_reward = 0

        for timestep in range(timesteps):
            
            agent.epsilon_greedy(env.get_valid_actions(agent.current_state))

            agent.set_prime_state(env.get_prime_state())

            agent.set_position(env.convert_state_to_pos(agent.current_state))
            env.check_diamond_mined(agent.current_state, agent.prev_state)

            if env.check_terminal(agent.current_state, agent.prev_state):
                agent.update_q()
                print(f'{name} - Terminal Reached {timestep}')
                break
            
            agent.update_q()

        ep_rewards.append(agent.total_reward)
        ep_steps.append(timestep)

    f = open('training_debug.txt', 'w')

    for state in enumerate(agent.Q.tolist()):
        f.write(f'\n{state}\n')

    f.close()

    with open(f'./agents/{name}/{name}-agent.pkl', 'wb') as file:
        pickle.dump(agent, file)

    return ep_rewards, ep_steps


def test_agent(name):
    images = []
    agent_rewards = []
    
    print(f'loading agent {name}')
    with open(f'./agents/{name}/{name}-agent.pkl', 'rb') as file:
        agent = pickle.load(file)

    print(f'Starting Test:')
    env.reset_env()
    agent.set_state(env.start_state)
    agent.set_position(agent.current_state)
    agent.total_reward = 0
    steps = 0

    images.append(env.get_grid(agent.position))

    while True:

        agent.epsilon_test(env.get_valid_actions(agent.current_state))

        agent_rewards.append(agent.total_reward)

        agent.set_prime_state(env.get_prime_state())
        agent.set_position(agent.current_state)

        env.check_diamond_mined(agent.current_state, agent.prev_state)

        steps += 1

        images.append(env.get_grid(agent.position))

        if env.check_terminal(agent.current_state, agent.prev_state):
            print(f"{name} - Terminal Reached: {steps} steps")
            print(agent.current_state)
            break

    print(f'{name} - Total reward: {agent.total_reward}')

    visualisePath(images, agent_rewards, name)

    plt.plot(agent.rewards)
    plt.show()

# Base Model Hyperparameters
# alpha = 0.8
# gamma = 0.9
# epsilon = 0.5

# Inference Model Hyperparameters
alpha = 0.6
gamma = 0.3
epsilon = 0.3

episodes = 1000
timesteps = 1000

run_config = {
    "alpha": 0.8,
    "gamma": 0.9,
    "epsilon": 0.5,
    "alphas": [0.9, 0.6, 0.3],
    "gammas": [0.9, 0.6, 0.3],
    "epsilons": [0.9, 0.6, 0.3]
}

def compare_params(run_config, eps, ts):
    alpha_results = {}
    gamma_results = {}
    epsilon_results = {}

    for alpha in run_config['alphas']:
        ep_train_rewards, ep_steps = train_agent(alpha, run_config["gamma"], run_config["epsilon"], episodes, timesteps)
        alpha_results[f'reward:{alpha}'] = ep_train_rewards
        alpha_results[f'steps:{alpha}'] = ep_steps

    for gamma in run_config['gammas']:
        ep_train_rewards, ep_steps = train_agent(run_config["alpha"], gamma, run_config["epsilon"], episodes, timesteps)
        gamma_results[f'reward:{gamma}'] = ep_train_rewards
        gamma_results[f'steps:{gamma}'] = ep_steps

    for epsilon in run_config['epsilons']:
        ep_train_rewards, ep_steps = train_agent(run_config["alpha"], run_config["gamma"], epsilon, episodes, timesteps)
        epsilon_results[f'reward:{epsilon}'] = ep_train_rewards
        epsilon_results[f'steps:{epsilon}'] = ep_steps

    with open(os.path.join(os.getcwd(), 'basic_task/results', "alpha_results.json"), 'w') as alpha_results_file:
        json.dump(alpha_results, alpha_results_file)

    with open(os.path.join(os.getcwd(), 'basic_task/results', "gamma_results.json"), 'w') as gamma_results_file:
        json.dump(gamma_results, gamma_results_file)

    with open(os.path.join(os.getcwd(), 'basic_task/results', "epsilon_results.json"), 'w') as epsilon_results_file:
        json.dump(epsilon_results, epsilon_results_file)

# compare_params(run_config, episodes, timesteps)

def plot_results(value, target, title):

    plt.title(title)
    plt.xlabel('episodes')
    plt.ylabel(target)
    colors = ['red', 'green', 'blue', 'olive', 'purple']
    color_id = 0

    window_size = 100
    vals_window = deque([])
    weighted_line = []
    for ind, val in enumerate(value):
        if ind < 100:
            vals_window.append(val)
            weighted_line.append(0)
        elif ind >= len(value):
            break
        else:
            vals_window.popleft()
            vals_window.append(val)
            weighted_line.append(sum(vals_window)/window_size)

    plt.plot(value)
    plt.plot(weighted_line, label=f'Average', color=colors[color_id])
    color_id += 1
    plt.legend()
    plt.show()

# ep_train_rewards, ep_steps = train_agent(alpha, gamma, epsilon, episodes, timesteps)

test_agent('a06g03e03')

# print(f'{ep_train_rewards}, \n{len(ep_train_rewards)}')
# print(f'{ep_steps}, \n{len(ep_steps)}')

# plot_results(ep_train_rewards, "reward", "Training Rewards")
# plot_results(ep_steps, "steps", "Training Steps")