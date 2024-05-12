import sys
import numpy as np
import copy
from itertools import product
from math import floor

np.set_printoptions(threshold=sys.maxsize)

class Environment():
    def __init__(self, 
            grid_x, grid_y, 
            diamond_collected_reward, corrosive_fume_reward, exit_reward, 
            start_exit_pos, wall_rail_pos, diamond_ore_pos, corrosive_fumes_pos,
            rails_mapping_pos):

        # env design params
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.amount_of_grid_states = grid_x*grid_y

        # env positions
        self.start_exit_pos = start_exit_pos
        self.wall_rail_pos = wall_rail_pos
        self.diamond_ore_pos = diamond_ore_pos
        self.corrosive_fumes_pos = corrosive_fumes_pos
        self.rails_mapping_pos = rails_mapping_pos

        # converted to states
        self.start_state = self.convert_pos_to_state(start_exit_pos)
        self.wall_rail_states = self.convert_pos_to_state(wall_rail_pos)
        self.diamond_ore_states = self.convert_pos_to_state(diamond_ore_pos)
        print(self.diamond_ore_states)
        self.corrosive_fumes_states = self.convert_pos_to_state(corrosive_fumes_pos)

        self.rails_mapping_states = []
        for mapping in self.rails_mapping_pos:
            self.rails_mapping_states.append(self.convert_pos_to_state(mapping))

        self.rail_starts = []
        for rail_start in self.rails_mapping_states:
            self.rail_starts.append(rail_start[0])


        self.diamonds_collected = []
        for i in range(len(self.diamond_ore_states)):
            self.diamonds_collected.append(False)

        self.prime_states = []
        self.prime_states = self.generate_prime_state()

        # rewards
        self.diamond_collected_reward = diamond_collected_reward
        self.corrosive_fume_reward = corrosive_fume_reward
        self.exit_reward = exit_reward

        self.all_actions = np.array(range(0, 5))
        self.all_states = np.array(range(0, self.amount_of_grid_states+len(self.diamond_ore_pos)+1))

        self.R = copy.deepcopy(self.get_R_matrix())

    def convert_pos_to_state(self, pos_arr):
        res = []
        for pos in pos_arr:
            res.append(pos[0]+(pos[1]*self.grid_x))
        #print(res)
        if len(res) == 1:
            return res[0]
        else:
            return res
        
    def convert_state_to_pos(self, state):
        position = [None, None]
        position[0] = floor(state/self.grid_x)
        position[1] = state % self.grid_x
        
        return position

    def generate_prime_state(self):
        
        prime_states = []
        
        prime_states = list(map(list, product([False, True], repeat = len(self.diamonds_collected))))

        print(f'Prime States: {prime_states}')

        if [True, False, False] in prime_states:
            print(f'Idx: {prime_states.index([True, False, False])}')

        return prime_states

    def get_R_matrix(self):
        
        R_mult_dim = []
        R =  np.empty((len(self.all_states),len(self.all_states),))
        R[:] = np.nan

        if len(self.diamond_ore_states) < 3:

            for index in range(0, len(self.diamond_ore_states) ** 2):

                R = self.add_rewards(self.generate_init_r_matrix(R), index)
                R_mult_dim.append(R)

        else:
            for index in range(0, (len(self.diamond_ore_states) ** 2) - 1):

                R = self.add_rewards(self.generate_init_r_matrix(R), index)
                R_mult_dim.append(copy.deepcopy(R))

        
        return R_mult_dim

    def check_legal(self, temp_state, change):
        
        if temp_state > self.amount_of_grid_states - 1 and temp_state < self.amount_of_grid_states + len(self.diamond_ore_states): 
            state = self.diamond_ore_states[temp_state - self.amount_of_grid_states]

        elif temp_state == self.amount_of_grid_states + len(self.diamond_ore_states):
            state = self.start_state

        else:
            state = temp_state
            
        new_state = state + change
        if state >= self.amount_of_grid_states and state < self.amount_of_grid_states + len(self.diamond_ore_states):
            new_state = self.diamond_ore_states[state - self.amount_of_grid_states] + change

        if (new_state < 0) or (new_state > self.amount_of_grid_states-1):
            return state
        elif (change == -1 and state % self.grid_x == 0) or (change == 1 and state % self.grid_x == (self.grid_x - 1)):
            return state
        elif new_state in self.wall_rail_states:
            return state
        else:
            return new_state

    def check_ore(self, state):
        if state in self.diamond_ore_states:
            return [state, self.amount_of_grid_states + self.diamond_ore_states.index(state)]
        else:
            return state

    def mine_action_map(self, state):
        if state >= self.amount_of_grid_states and state < self.amount_of_grid_states + len(self.diamond_ore_states):
            ind = state - self.amount_of_grid_states
            return self.diamond_ore_states[ind]
        else:
            return state

    def empty_mine_action(self, state, temp_r_matrix):
        if state < self.amount_of_grid_states:
            temp_r_matrix[state][state] = 0
        return temp_r_matrix

    def check_rails(self, state):
        for mapping in self.rails_mapping_states:
            if state == mapping[0]:
                return mapping[1]
            else:
                continue
        return state

    def generate_init_r_matrix(self, nan_r):
        # 0 = up, 1 = right, 2 = down, 3 = left, 4 = mine
        for state in self.all_states:
            if (state not in self.wall_rail_states):
                if state not in self.rail_starts:
                    for action in self.all_actions:
                        match action:
                            case 0:
                                border_check = self.check_legal(state, -self.grid_x)
                                if border_check != state:
                                    rail_check = self.check_rails(border_check)
                                    new_states = self.check_ore(rail_check)
                                else:
                                    new_states = border_check
                            case 1:
                                border_check = self.check_legal(state, 1)
                                if border_check != state:
                                    rail_check = self.check_rails(border_check)
                                    new_states = self.check_ore(rail_check)
                                else:
                                    new_states = border_check
                            case 2:
                                border_check = self.check_legal(state, self.grid_x)
                                if border_check != state:
                                    rail_check = self.check_rails(border_check)
                                    new_states = self.check_ore(rail_check)
                                else:
                                    new_states = border_check
                            case 3:
                                border_check = self.check_legal(state, -1)
                                if border_check != state:
                                    rail_check = self.check_rails(border_check)
                                    new_states = self.check_ore(rail_check)
                                else:
                                    new_states = border_check
                            case 4:
                                new_states = self.mine_action_map(state)

                        if type(new_states) == list:
                            for new_state in new_states:
                                if new_state != state:
                                    nan_r[state][new_state] = 0
                                else:
                                    continue
                        else:
                            if new_states != state:
                                nan_r[state][new_states] = 0

                        nan_r = self.empty_mine_action(state, nan_r)

            else:
                continue
            
        return nan_r

    def add_rewards(self, init_r_matrix, prime_state):
        for index, diamond_ore_state in enumerate(self.diamond_ore_states):
            
            if self.prime_states[prime_state][index] == self.diamonds_collected[index] and not self.diamonds_collected[index]:
                
                init_r_matrix[self.amount_of_grid_states+index][diamond_ore_state] = self.diamond_collected_reward

        for starting_state, reward in enumerate(init_r_matrix):
            valid_actions = np.where(~np.isnan(init_r_matrix[starting_state]))[0]
            for valid_action in valid_actions:
                if valid_action in self.corrosive_fumes_states:
                    init_r_matrix[starting_state][valid_action] = self.corrosive_fume_reward

        if prime_state == len(self.prime_states) - 1:

            for starting_state, reward in enumerate(init_r_matrix):

                valid_actions = np.where(~np.isnan(init_r_matrix[starting_state]))[0]
                for valid_action in valid_actions:
                    if valid_action == self.start_state and starting_state != self.start_state:
                        init_r_matrix[starting_state][-1] = 0
                        init_r_matrix[self.amount_of_grid_states + len(self.diamond_ore_states)][valid_action] = self.exit_reward

        for starting_state, reward in enumerate(init_r_matrix):
            valid_actions = np.where(~np.isnan(init_r_matrix[starting_state]))[0]
            for valid_action in valid_actions:
                if valid_action == starting_state:
                    
                    init_r_matrix[starting_state][valid_action] = -20

        return init_r_matrix

    def get_valid_actions(self, current_state):
        
        prime_state = self.prime_states.index(self.diamonds_collected)

        valid_actions = np.where(~np.isnan(self.R[prime_state][current_state]))[0]
        
        for index, diamond_ore in enumerate(self.diamond_ore_states):
            if diamond_ore in valid_actions:

                if self.diamonds_collected[index]:
                    
                    valid_actions = np.delete(valid_actions, np.where(valid_actions == self.amount_of_grid_states + index))

                elif current_state > self.amount_of_grid_states-1 and current_state < self.amount_of_grid_states + len(self.diamond_ore_states):
                    
                    continue

                else:
                    valid_actions = np.delete(valid_actions, np.where(valid_actions == diamond_ore))

        if self.start_state in valid_actions:
            if False in self.diamonds_collected:
                valid_actions = np.delete(valid_actions, np.where(valid_actions == self.amount_of_grid_states+len(self.diamond_ore_pos)))

        return valid_actions

    def check_diamond_mined(self, agent_current_state, agent_prev_state):
        
        if (agent_current_state in self.diamond_ore_states) and (agent_prev_state == self.amount_of_grid_states + self.diamond_ore_states.index(agent_current_state)):
            
            self.diamonds_collected[self.diamond_ore_states.index(agent_current_state)] = True

    def check_terminal(self, current_state, prev_state):
        
        if self.get_prime_state() == len(self.prime_states) -1 and current_state == self.start_state and prev_state == self.amount_of_grid_states + len(self.diamond_ore_states):
            
            return True
        
        else:
            return False

    def reset_env(self):
        
        self.diamonds_collected = [False, False, False]
        self.diamond_ore_states = self.convert_pos_to_state(self.diamond_ore_pos)

    def get_grid(self, agent_state):
        empty_grid = np.zeros((self.grid_x, self.grid_y))

        agent_pos = self.convert_state_to_pos(agent_state)

        if agent_state > self.amount_of_grid_states - 1:

            if agent_state < self.amount_of_grid_states + len(self.diamond_ore_states):
                agent_pos = self.convert_state_to_pos(self.diamond_ore_states[agent_state - self.amount_of_grid_states])
                empty_grid[agent_pos[0], agent_pos[1]] += 50
            
            else:
                agent_pos[0] = self.start_exit_pos[0][1]
                agent_pos[1] = self.start_exit_pos[0][0]

                empty_grid[agent_pos[0], agent_pos[1]] += 0

        if False not in self.prime_states[self.get_prime_state()] and agent_state == self.start_state:

            agent_pos[0] = self.start_exit_pos[0][1]
            agent_pos[1] = self.start_exit_pos[0][0]

            empty_grid[agent_pos[0], agent_pos[1]] += 100

        for wall_rail in self.wall_rail_pos:
            empty_grid[wall_rail[1], wall_rail[0]] += -5

        for rail_start_state in self.rail_starts:
            rail_start = self.convert_state_to_pos(rail_start_state)
            empty_grid[rail_start[0], rail_start[1]] += 5

        for fumes in self.corrosive_fumes_pos:
            empty_grid[fumes[1], fumes[0]] += -30

        for index, diamond in enumerate(self.diamond_ore_pos):
            if self.diamonds_collected[index]:
                continue
            else:
                empty_grid[diamond[1], diamond[0]] += 50

        if False not in self.diamonds_collected:
            empty_grid[self.start_exit_pos[0][1], self.start_exit_pos[0][0]] += 100
        
        empty_grid[agent_pos[0], agent_pos[1]] += 1000

        return empty_grid
    
    def generate_R_matrix(self):
        
        f = open('debug.txt', 'w')
        
        for index, state in enumerate(self.R):
            
            f.write(f'\nfrom state: {index}')

            for idx, s in enumerate(state):
            
                f.write(f'\nfrom state: [{index}][{idx}]')

                i = 0
                while i < self.amount_of_grid_states:
                    f.write(f'\n{np.array2string(s[i:i + self.grid_x])}')
                    i+= self.grid_x
                    
                f.write(f'{np.array2string(s[self.amount_of_grid_states: self.amount_of_grid_states + len(self.diamond_ore_states)])}')
                f.write(f'{np.array2string(s[self.amount_of_grid_states + len(self.diamond_ore_states): self.amount_of_grid_states + len(self.diamond_ore_states) + 1 ])}\n')

        f.close()

    def get_prime_state(self):

        return self.prime_states.index(self.diamonds_collected)
