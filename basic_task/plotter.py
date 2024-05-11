from matplotlib import pyplot as plt
import json
import os

def plot_results(results_target_file, target):
    with open(os.path.join(os.getcwd(), 'results', f'{results_target_file}.json'), 'r') as file:
        data = json.load(file)
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    colors = ['red', 'green', 'blue', 'olive', 'purple']
    color_id = 0
    for key in data:
        if target == key.split(":")[0]:
            plt.plot(data[key], label=f'{key}', color=colors[color_id])
            color_id += 1
    plt.legend()
    plt.show()

plot_results("alpha_results", "reward")
plot_results("alpha_results", "steps")
# plot_results("gamma_results", "gamma")
# plot_results("epsilon_results", "ep_decay")