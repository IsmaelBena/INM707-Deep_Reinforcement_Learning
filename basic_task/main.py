import numpy as np

from environment import Environment
from agent import Agent

grid_x = 8  # Grid size
grid_y = 8
diamond_collected_reward = 50
corrosive_fume_reward = -30
exit_reward = 100
start_exit_pos = [[7, 3]]
wall_rail_pos = [[0, 1], [1, 1], [2, 1], [4, 1], [5, 1], [2, 2], [1, 3], [2, 3], [4, 4], [5, 4], [6, 4], [2, 6], [4, 6], [5, 6], [6, 6], [5, 0], [6, 0], [1, 7]]
diamond_ore_pos = [[0,0], [0, 5], [5, 5]]
corrosive_fumes_pos = [[1, 2], [6, 5], [0, 7]]
rails_mapping_pos = [[[6, 1], [3, 0]], [[4, 0], [6, 2]], [[2, 7], [1, 5]], [[1, 6], [3, 7]]]

alpha = 1
gamma = 0.8
epsilon = 0.1

env = Environment(grid_x, grid_y, diamond_collected_reward, corrosive_fume_reward, exit_reward, start_exit_pos, wall_rail_pos, diamond_ore_pos, corrosive_fumes_pos, rails_mapping_pos)

agent = Agent(alpha, gamma, epsilon, env.R)

for episode in range(1000):
    print(f'Starting Episode {episode}:')
    agent.set_position(env.start_state)
    
    for timestep in range(500000):
        agent.take_action(env.get_valid_actions(agent.current_state))
        agent.update_q()
        env.check_diamond_mined(agent.current_state, agent.prev_state)

        if env.check_terminal(agent.current_state):
            break
    
    #print(f'Episode {episode} finished with Q value: \n{agent.Q.round(1)}')

# print(f'Final Q Matrix: \n {agent.Q.round(1)}')

f = open('training_debug.txt', 'w')

for state in enumerate(agent.Q.tolist()):
    f.write(f'\n{state}')

f.close()