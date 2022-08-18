import matplotlib.pyplot as plt
import numpy as np
import re
import os
import matplotlib as mpl
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
sns.set()
from scipy.stats import entropy

def listdirs(path):
    list = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sorted(list)


def moving_average(data, M=5):
    temp_data = np.concatenate((np.repeat(data[0], int(M / 2)), data, np.repeat(data[-1], int(M / 2))))
    data = np.convolve(temp_data, np.ones(M) / M, mode='valid')
    return data


listdirs_ml10 = listdirs('logs/logs_ML10Env-v2')
seed_list = [10]
date = '17:08'
#seed_list = [17]
dir_ml10=[]
for seed in seed_list:
    for dir in listdirs_ml10:
        if "varibad_"+str(seed)+'__'+date in dir:
            dir_ml10.append(dir)
print(dir_ml10)

#iter_list = [0,1,2,3,4,5,6]
iter_list = [0]
task_successes = np.zeros((len(iter_list), 15,len(seed_list))) #first 10 rows are train
frame_list = []
iter_list_ = []

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
        iter_list_.append(df['iter'][iter])
print(task_successes)

palette = sns.color_palette()
color_list = [palette[0]]*10 + [palette[1]]*5
palette=sns.color_palette("hls", 15)

for seed in range(len(seed_list)):
    for iter in range(len(iter_list)):
        train_success = np.mean(task_successes[iter, 0:10, seed])
        test_success = np.mean(task_successes[iter, 10:, seed])

        fig, axes = plt.subplots(4,1, figsize=(8,10), gridspec_kw={'height_ratios': [1, 1, 1, 1]})

        data_1 = np.load(dir_ml10[seed] + '/' + str(int(iter_list_[iter])) + '/probs.npy')
        print(data_1)

        data_ = np.load(dir_ml10[seed]+'/'+str(int(iter_list_[iter]))+'/episode_probs_array.npy')
        #(15, total_parametric_num, self.args.max_rollouts_per_task, self.envs._max_episode_steps, self.args.vae_mixture_num)
        #(15,50,10,500,10)
        print(np.shape(data_))
        print(data_)

        prametric_num = np.shape(data_)[1]
        max_rollouts_per_task = np.shape(data_)[2]
        max_episode_steps = np.shape(data_)[3]
        class_num = np.shape(data_)[4]

        step_list = [0, 50, 499, 4999]
        for i in range(4):
            heatmap_data = np.zeros((class_num, 15), dtype=int)
            heatmap_data = data_[:,:, step_list[i]//max_episode_steps, step_list[i]%max_episode_steps,:]
            heatmap_data = np.mean(heatmap_data, axis=1)
            sns.heatmap(np.flip(np.transpose(heatmap_data),0), ax=axes[i], vmin=0, vmax=1.0, annot=True, fmt='.2f', cmap = 'viridis', cbar=False, xticklabels=range(1, 16),yticklabels=range(10, 0, -1))
            axes[i].set_ylabel('Mean_subtasks \n y')
        plt.show()
        plt.close()










