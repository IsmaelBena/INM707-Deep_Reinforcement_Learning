import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import pickle
import os

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

# alpha = 0.8
# gamma = 0.9
# epsilon = 0.5

env = Environment(grid_x, grid_y, diamond_collected_reward, corrosive_fume_reward, exit_reward, start_exit_pos, wall_rail_pos, diamond_ore_pos, corrosive_fumes_pos, rails_mapping_pos)
env.generate_R_matrix()

# agent = Agent(alpha, gamma, epsilon, env.R)

# images = []
# frames = []
# agent_rewards = []

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

# def plotState(grid):
#     bounds = [-31, -6, -1, 1, 6, 51, 101, 1001, 1101, 1201]
#     cmap = mpl.colors.ListedColormap(['purple','brown','white', 'red', 'blue', 'yellow', 'black', 'orange', 'green'])
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#     img = plt.imshow(grid, interpolation='nearest', cmap = cmap,norm=norm, animated = True)

#     images.append(grid)
#     # make a color bar
#     #pyplot.colorbar(img,cmap=cmap, norm=norm, boundaries=bounds,ticks=[-5,0,5])
#     #pyplot.show()

#     #images.append([img])
#     #np.append(images, img)

#     #pyplot.show()

def visualisePath(images, agent_rewards, name):
    
    print(f'Generating Path... {len(images)} frames')

    fig = plt.figure()
    frames = []

    bounds = [-31, -6, -1, 1, 6, 51, 101, 1001, 1101, 1201]
    cmap = mpl.colors.ListedColormap(['purple','brown','white', 'red', 'blue', 'yellow', 'black', 'orange', 'green'])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    #img = pyplot.imshow(grid, interpolation='nearest', cmap = cmap,norm=norm, animated = True)
    #reward = agent.total_reward

    for idx, img in enumerate(images):
        
        frames.append([plt.imshow(img, interpolation='nearest', cmap = cmap,norm=norm, animated = True), plt.annotate(f'Step: {idx}, R: {agent_rewards[idx - 1]}', (0, 0))])

    
    ani = animation.ArtistAnimation(fig, frames)
    ani.save(f'./agents/{name}/path.mp4')

    #pyplot.show()

def plot_reward(rewards, name):

    img = plt.plot(rewards)

    plt.savefig(f'./agents/{name}/rewards.png')


def train_agent(alpha, gamma, epsilon, episodes, timesteps):
    name = f'a{alpha}g{gamma}e{epsilon}'.replace('.', '')
    decay = epsilon/timesteps
    
    if not os.path.exists(f'./agents/{name}'):
        os.makedirs(f'./agents/{name}')

    agent = Agent(alpha, gamma, epsilon, decay, env.R, name)
    
    images = []
    agent_rewards = []

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

            #agent.update_env(env.R)

            if env.check_terminal(agent.current_state, agent.prev_state):
                agent.update_q()
                print(f'{name} - Terminal Reached {timestep}')
                break
            
            agent.update_q()
            
        #print(f'Total reward: {agent.total_reward}')

        #print(env.moved_off_mine)
        
        #print(f'Episode {episode} finished with Q value: \n{agent.Q.round(1)}')

    # print(f'Final Q Matrix: \n {agent.Q.round(1)}')

    f = open('training_debug.txt', 'w')

    for state in enumerate(agent.Q.tolist()):
        f.write(f'\n{state}\n')

    f.close()

    plot_reward(agent.rewards, name)

    with open(f'./agents/{name}/{name}-agent.pkl', 'wb') as file:
        pickle.dump(agent, file)


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
    #plotState(env.get_grid(agent.position))

    while True:
        #print(env.get_valid_actions(agent.current_state))

        agent.epsilon_test(env.get_valid_actions(agent.current_state))

        agent_rewards.append(agent.total_reward)

        agent.set_prime_state(env.get_prime_state())
        agent.set_position(agent.current_state)

        #print(env.get_prime_state())

        env.check_diamond_mined(agent.current_state, agent.prev_state)

        steps += 1

        images.append(env.get_grid(agent.position))
        #plotState(env.get_grid(agent.position))

        #agent.update_env(env.R)
        
        #print(agent.current_state)

        if env.check_terminal(agent.current_state, agent.prev_state):
            print(f"{name} - Terminal Reached: {steps} steps")
            print(agent.current_state)
            break

    print(f'{name} - Total reward: {agent.total_reward}')

    #env.generate_R_matrix()

    visualisePath(images, agent_rewards, name)

    #plt.plot(agent.rewards)
    #plt.show()


# alpha = 0.8
# gamma = 0.9
# epsilon = 0.5

#     name = f'a{alpha}g{gamma}e{epsilon}'.replace('.', '')

alphas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
gammas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
epsilons = [1, 0.75, 0.5, 0.25]

episodes = 500
timesteps = 10000

for alpha in alphas:
    for gamma in gammas:
        for epsilon in epsilons:
            train_agent(alpha, gamma, epsilon, episodes, timesteps)
            test_agent(f'a{alpha}g{gamma}e{epsilon}'.replace('.', ''))