import sys
import numpy as np
from math import floor

np.set_printoptions(threshold=sys.maxsize)

class Environment():
    def __init__(self, 
            grid_x, grid_y, 
            diamond_collected_reward, corrosive_fume_reward, exit_reward, 
            start_exit_pos, wall_rail_pos, diamond_ore_pos, corrosive_fumes_pos,
            rails_mapping_pos):

        ####################### Debugging
        #self.moved_off_mine = 0


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


        # rewards
        self.diamond_collected_reward = diamond_collected_reward
        self.corrosive_fume_reward = corrosive_fume_reward
        self.exit_reward = exit_reward

        self.all_actions = np.array(range(0, 5))
        self.all_states = np.array(range(0, self.amount_of_grid_states+len(self.diamond_ore_pos)+1))

        self.R =  np.empty((len(self.all_states),len(self.all_states),))
        self.R[:] = np.nan

        self.R = self.add_rewards(self.generate_init_r_matrix(self.R))

        f = open('debug.txt', 'w')

        for index, state in enumerate(self.R):
            f.write(f'\nfrom state: {index}')

            i = 0
            while i < self.amount_of_grid_states:
                f.write(f'\n{np.array2string(state[i:i + self.grid_x])}')
                i+= self.grid_x
                

            f.write(f'{np.array2string(state[self.amount_of_grid_states: self.amount_of_grid_states + len(self.diamond_ore_states)])}')
            f.write(f'{np.array2string(state[self.amount_of_grid_states + len(self.diamond_ore_states): self.amount_of_grid_states + len(self.diamond_ore_states) + 1 ])}\n')

        f.close()

        #print(self.R.shape)

        self.diamonds_collected = [False, False, False]


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

    def check_legal(self, temp_state, change):
        
        if temp_state > self.amount_of_grid_states - 1:
            state = self.diamond_ore_states[temp_state - self.amount_of_grid_states]
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
            if state == self.amount_of_grid_states + len(self.diamond_ore_states):
                continue
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

    def add_rewards(self, init_r_matrix):
        for index, diamond_ore_state in enumerate(self.diamond_ore_states):
            init_r_matrix[self.amount_of_grid_states+index][diamond_ore_state] = self.diamond_collected_reward

        for starting_state, reward in enumerate(init_r_matrix):
            #print(np.where(~np.isnan(init_r_matrix[starting_state]))[0])
            valid_actions = np.where(~np.isnan(init_r_matrix[starting_state]))[0]
            for valid_action in valid_actions:
                #print(valid_action)
                if valid_action in self.corrosive_fumes_states:
                    init_r_matrix[starting_state][valid_action] = self.corrosive_fume_reward

        for starting_state, reward in enumerate(init_r_matrix):
            #print(np.where(~np.isnan(init_r_matrix[starting_state]))[0])
            valid_actions = np.where(~np.isnan(init_r_matrix[starting_state]))[0]
            for valid_action in valid_actions:
                #print(valid_action)
                if valid_action == self.start_state:
                    init_r_matrix[starting_state][-1] = self.exit_reward

        return init_r_matrix

    def get_valid_actions(self, current_state):
        #print(f'curr state {current_state}')
        #print(f'np where {np.where(~np.isnan(self.R[current_state]))}')
        valid_actions = np.where(~np.isnan(self.R[current_state]))[0]
        #print(f'valid actions {valid_actions}')
        for index, diamond_ore in enumerate(self.diamond_ore_states):
            if diamond_ore in valid_actions:
                #print(f'diamond ore {index} at {diamond_ore} status: {self.diamonds_collected[index]}')
                if self.diamonds_collected[index]:
                    #print(f'currently at: {current_state}\nActions: {valid_actions}\nRemoving: {self.amount_of_grid_states + index}')
                    valid_actions = np.delete(valid_actions, np.where(valid_actions == self.amount_of_grid_states + index))
                    #print(f'After removing: {valid_actions}')
                elif current_state > self.amount_of_grid_states-1 and current_state < self.amount_of_grid_states + len(self.diamond_ore_states):
                    continue
                else:
                    #print(f'currently at: {current_state}\nActions: {valid_actions}\nRemoving: {diamond_ore}')
                    valid_actions = np.delete(valid_actions, np.where(valid_actions == diamond_ore))
                    #print(f'After removing: {valid_actions}')

        if self.start_state in valid_actions:
            if False in self.diamonds_collected:
                valid_actions = np.delete(valid_actions, np.where(valid_actions == self.amount_of_grid_states+len(self.diamond_ore_pos)))
            else:
                valid_actions = np.delete(valid_actions, np.where(valid_actions == self.start_state))

        return valid_actions

    def check_diamond_mined(self, agent_current_state, agent_prev_state):
        #print(f'from {agent_prev_state} to {agent_current_state}')
        if (agent_current_state in self.diamond_ore_states) and (agent_prev_state == self.amount_of_grid_states + self.diamond_ore_states.index(agent_current_state)):
            self.diamonds_collected[self.diamond_ore_states.index(agent_current_state)] = True
        #    print(f'Diamonds mined: {self.diamonds_collected}')
        #     print('Diamond Mined')
        # elif agent_current_state == 0 or agent_current_state == 40 or agent_current_state == 45:
        #     print('empty ore deposit?')
        # elif agent_prev_state == 64 or agent_prev_state == 65 or agent_prev_state == 66:
        #     self.moved_off_mine += 1

        #if (agent_current_state == 0 or agent_current_state == 40 or agent_current_state == 45) and (agent_prev_state == 64 or agent_prev_state == 65 or agent_prev_state == 66):
            #print('mining not check properly')

    def check_terminal(self, state):
        if state == self.amount_of_grid_states + len(self.diamond_ore_states):
            return True
        else:
            return False

    def reset_env(self):
        self.diamonds_collected = [False, False, False]



    def get_grid(self, agent_pos):
        empty_grid = np.zeros((self.grid_x, self.grid_y))
        print(empty_grid)

        print(agent_pos)

        if agent_pos[0] > self.grid_x - 1:
            print("mine")
            agent_pos[0] = agent_pos[0] - self.grid_x
            print(agent_pos)
        
        elif agent_pos[1] > self.grid_y - 1:
            print("mine y")
            agent_pos[1] = agent_pos[1] - self.grid_y

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
            empty_grid[self.start_exit_pos[1], self.start_exit_pos[0]] += 100
        
        empty_grid[agent_pos[0], agent_pos[1]] += 1000

        print(empty_grid)

        return empty_grid