import matplotlib.pyplot as plt
import numpy as np
import re
import os
import matplotlib as mpl
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns
sns.set()

def listdirs(path):
    list = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sorted(list)


def moving_average(data, M=5):
    temp_data = np.concatenate((np.repeat(data[0], int(M / 2)), data, np.repeat(data[-1], int(M / 2))))
    data = np.convolve(temp_data, np.ones(M) / M, mode='valid')
    return data


dir_ml10 = listdirs('logs/logs_ML10Env-v2')
dir_ml10 = [dir_ml10[0]]+dir_ml10[9:]


task_successes = np.zeros((2, 15,8)) #first 10 rows are train
for iter in range(2):
    for seed in range(8):
        dir = dir_ml10[seed]
        df = pd.read_csv(dir+'/log_vis.csv')
        keys = df.keys()
        for task in range(15):
            task_successes[iter, task, seed] = df[keys[15+task]][iter]

palette = sns.color_palette()
color_list = [palette[0]]*10 + [palette[1]]*5

data_mean = np.mean(task_successes, axis=2)
data_std = np.std(task_successes, axis=2)
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

print(data_mean[0])
print(data_std[0])

ax1.set_title('VariBAD Success Rate at 0 steps (8 seeds)')
ax1.bar(range(1,16), 100*data_mean[0], yerr=100*data_std[0], alpha=0.5, color = color_list)
ax1.set_xticks(range(1,16))
ax1.set_ylim([0, 100])
ax1.set_ylabel('Success Rate (%)', fontsize=15)

ax2.set_title('VariBAD Success Rate at 25M steps (8 seeds)')
ax2.bar(range(1,16), 100*data_mean[1], yerr=100*data_std[1], alpha=0.5, color = color_list)
ax2.set_xticks(range(1,16))
ax2.set_ylim([0, 100])
ax2.set_xlabel('Task Index', fontsize=15)
ax2.set_ylabel('Success Rate (%)', fontsize=15)

legend_elements = [Patch(facecolor=palette[0], edgecolor=palette[0], alpha=0.5, label='Train Tasks'),
                   Patch(facecolor=palette[1], edgecolor=palette[1], alpha=0.5, label='Test Tasks')]
ax1.legend(handles = legend_elements, loc='upper left')

plt.subplots_adjust(hspace=0.4, wspace =0.05, bottom=0.15, top=0.86, left=0.15, right=0.97)
plt.savefig('plots/Histogram_' + '0' + '.png')
plt.show()
plt.close()