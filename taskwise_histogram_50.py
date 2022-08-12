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


listdirs_ml10 = listdirs('logs/logs_ML10Env-v2')
seed_list = range(10,18)
dir_ml10=[]
for seed in seed_list:
    for dir in listdirs_ml10:
        if "varibad_"+str(seed)+'_' in dir:
            dir_ml10.append(dir)
print(dir_ml10)

iter_list = []
for dir in dir_ml10:
    df = pd.read_csv(dir+'/log_eval.csv')
    iter_list.append(len(df))
min_iter = min(iter_list)
print(min_iter)
iter_list = range(min_iter)
task_successes = np.zeros((len(iter_list), 15,len(seed_list))) #first 10 rows are train
frame_list = []
for seed in range(len(seed_list)):
    dir = dir_ml10[seed]
    df = pd.read_csv(dir+'/log_eval.csv')
    keys = df.keys()
    iter_num = 0
    for iter in iter_list:
        for task in range(15):
            task_successes[iter_num, task, seed] = df[keys[17+task]][iter]
        iter_num+=1
        frame_list.append(df['frames'][iter])
print(task_successes)

palette = sns.color_palette()
color_list = [palette[0]]*10 + [palette[1]]*5

data_mean = np.mean(task_successes, axis=2)
data_std = np.std(task_successes, axis=2)
fig = plt.figure(figsize=(12,1+2.0*((len(iter_list)+2)//3)))
axes = []
for i in range(len(iter_list)):
    axes.append(fig.add_subplot((len(iter_list)+2)//3,3,i+1))

for i in range(len(iter_list)):
    if i==0:
        axes[i].set_title('VariBAD Success Rate at {:d}M steps ({:d} seeds)'.format(int(frame_list[i]//1e6),len(seed_list)))
    else:
        axes[i].set_title('at {:d}M steps'.format(int(frame_list[i] // 1e6)))
    axes[i].bar(range(1,16), 100*data_mean[i], yerr=100*data_std[i], alpha=0.5, color = color_list)
    axes[i].set_xticks(range(1,16))
    axes[i].set_ylim([0, 100])
    if i%3==0:
        axes[i].set_ylabel('Success Rate (%)')
    if i>len(iter_list)-len(iter_list)//3:
        axes[i].set_xlabel('Task Index')

legend_elements = [Patch(facecolor=palette[0], edgecolor=palette[0], alpha=0.5, label='Train Tasks'),
                   Patch(facecolor=palette[1], edgecolor=palette[1], alpha=0.5, label='Test Tasks')]
axes[0].legend(handles = legend_elements, loc='upper right')

plt.subplots_adjust(hspace=0.35, wspace =0.15, bottom=0.10, top=0.95, left=0.10, right=0.95)
plt.savefig('plots/Taskwise_histogram/Histogram_seed_{:d}to{:d}.png'.format(seed_list[0],seed_list[-1]))
plt.show()
plt.close()