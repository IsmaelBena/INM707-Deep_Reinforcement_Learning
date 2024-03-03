import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# env vars
grid_x = 8  # Grid size
grid_y = 8
grid_states = grid_x*grid_y

def convert_pos_to_state(pos_arr):
    res = []
    for pos in pos_arr:
        res.append(pos[0]+(pos[1]*8))
    print(res)
    return res

nan_pos = [[0, 1], [1, 1], [2, 1], [4, 1], [5, 1], [2, 2], [1, 3], [2, 3], [4, 4], [5, 4], [6, 4], [2, 6], [4, 6], [5, 6], [6, 6], [5, 0], [6, 0], [1, 7]]
nan_states = convert_pos_to_state(nan_pos)
diamond_ore_pos = [[0,0], [0, 5], [5, 5]]
diamond_ore_states = convert_pos_to_state(diamond_ore_pos)
corrosive_fumes_pos = [[1, 2], [6, 5], [0, 7]]
corrosive_fumes_states = convert_pos_to_state(corrosive_fumes_pos)

rails_mapping_pos = [[[6, 1], [3, 0]], [[4, 0], [6, 2]], [[2, 7], [1, 5]], [[1, 6], [3, 7]]]
rails_mapping_states = []
for mapping in rails_mapping_pos:
    rails_mapping_states.append(convert_pos_to_state(mapping))
print(rails_mapping_states)

all_actions = np.array(range(0, 5))
all_states = np.array(range(0, grid_states+len(diamond_ore_pos)))

R_matrix =  np.empty((len(all_states),len(all_states),))
R_matrix[:] = np.nan

print(f'Reward Matrix Shape: {np.shape(R_matrix)}')



def check_border(state, change):
    new_state = state + change
    if (new_state < 0) or (new_state > grid_states-1):
        return state
    elif (change == -1 and state % grid_x == 0) or (change == 1 and state % grid_x == (grid_x - 1)):
        return state
    elif new_state in nan_states:
        return state
    else:
        return new_state

def check_ore(state):
    if state in diamond_ore_states:
        return [state, grid_states + diamond_ore_states.index(state)]
    else:
        return state

def mine_action_map(state):
    if state >= grid_states:
        ind = state - grid_states
        return diamond_ore_states[ind]
    else:
        return state

def mine_map(state):
    if state >= grid_states:
        ind = state - grid_states
        return diamond_ore_states[ind]
    else:
        return state

def check_rails(state):
    for mapping in rails_mapping_states:
        if state == mapping[0]:
            return mapping[1]
        else:
            continue
    return state

def generate_init_r_matrix(nan_r):
    # 0 = up, 1 = right, 2 = down, 3 = left, 4 = mine
    for state in all_states:
        if (state not in nan_states):
            for action in all_actions:
                match action:
                    case 0:
                        border_check = check_border(mine_map(state), -grid_x)
                        if border_check != state:
                            rail_check = check_rails(border_check)
                            new_states = check_ore(rail_check)
                        else:
                            new_states = border_check
                    case 1:
                        border_check = check_border(mine_map(state), 1)
                        if border_check != state:
                            rail_check = check_rails(border_check)
                            new_states = check_ore(rail_check)
                        else:
                            new_states = border_check
                    case 2:
                        border_check = check_border(mine_map(state), grid_x)
                        if border_check != state:
                            rail_check = check_rails(border_check)
                            new_states = check_ore(rail_check)
                        else:
                            new_states = border_check
                    case 3:
                        border_check = check_border(mine_map(state), -1)
                        if border_check != state:
                            rail_check = check_rails(border_check)
                            new_states = check_ore(rail_check)
                        else:
                            new_states = border_check
                    case 4:
                        new_states = mine_action_map(state)

                if type(new_states) == list:
                    for new_state in new_states:
                        if new_state != state:
                            nan_r[state, new_state] = 0
                        else:
                            continue
                else:
                    if new_states != state:
                        nan_r[state, new_states] = 0
                    else:
                        continue
        else:
            continue
        
    return nan_r

R_matrix = generate_init_r_matrix(R_matrix)
#print(R_matrix)

f = open('debug.txt', 'w')

for index, state in enumerate(R_matrix):
    f.write(f'\nfrom state: {index}')

    i = 0
    while i < grid_states:
        f.write(f'\n{np.array2string(state[i:i + grid_x])}')
        i+= grid_x
        

    f.write(f'{np.array2string(state[grid_states: grid_states + len(diamond_ore_states)])}\n')

f.close()


#########################################################
    
starting_state = convert_pos_to_state([[7, 3]])

print(f'Starting State: {starting_state}')
print(f'Current Matrix: {R_matrix[starting_state]}')

valid_actions = np.where(~np.isnan(R_matrix[starting_state]))[1]
print(f'Valid Actions: {valid_actions}')