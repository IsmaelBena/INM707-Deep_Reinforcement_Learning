from matplotlib import pyplot as plt
import json
import os
from collections import deque

def plot_results(results_target_file, target, title):
    with open(os.path.join(os.getcwd(), 'basic_task/results', f'{results_target_file}.json'), 'r') as file:
        data = json.load(file)
    plt.title(title)
    plt.xlabel('episodes')
    plt.ylabel(target)
    colors = ['red', 'green', 'blue', 'olive', 'purple']
    color_id = 0
    for key in data:
        if target == key.split(":")[0]:
            window_size = 100
            vals_window = deque([])
            weighted_line = []
            for ind, val in enumerate(data[key]):
                if ind < 100:
                    vals_window.append(val)
                elif ind >= len(data[key]):
                    break
                else:
                    vals_window.popleft()
                    vals_window.append(val)
                    weighted_line.append(sum(vals_window)/window_size)

            plt.plot(weighted_line, label=f'{key[-3:]}', color=colors[color_id])
            color_id += 1
    plt.legend()
    plt.show()

plot_results("alpha_results", "reward", "Alpha Rewards")
plot_results("alpha_results", "steps", "Alpha Steps")
plot_results("gamma_results", "reward", "Gamma Rewards")
plot_results("gamma_results", "steps", "Gamma Steps")
plot_results("epsilon_results", "reward", "Epsilon Rewards")
plot_results("epsilon_results", "steps", "Epsilon Steps")