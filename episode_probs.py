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


listdirs_ml10 = listdirs('ml10-joint-log/vae10x_c5_d5_passprob_avgvae_adjustalpha/logs_ML10Env-v2/')
seed_list = [10, 11,12,13]
dir_ml10=[]
for seed in seed_list:
    for dir in listdirs_ml10:
        if "varibad_"+str(seed)+'_' in dir:
            dir_ml10.append(dir)
print(dir_ml10)

#iter_list = [0,1,2,3,4,5,6]
iter_list = [40]
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

        fig, axes = plt.subplots(4,2, figsize=(16,10))

        data_1 = np.load(dir_ml10[seed] + '/' + str(int(iter_list_[iter])) + '/probs.npy')
        #print(data_1)

        data_ = np.load(dir_ml10[seed]+'/'+str(int(iter_list_[iter]))+'/episode_probs_array.npy')
        #(15, total_parametric_num, self.args.max_rollouts_per_task, self.envs._max_episode_steps, self.args.vae_mixture_num)
        #(15,50,10,500,10)
        print(np.shape(data_))
        #print(data_)

        parametric_num = np.shape(data_)[1]
        max_rollouts_per_task = np.shape(data_)[2]
        max_episode_steps = np.shape(data_)[3]
        class_num = np.shape(data_)[4]

        step_list = [0, 0, 1, 2, 10, 49, 499, 4999]

        axes[0,0].bar(range(15), 100 * task_successes[iter,:,seed], alpha=0.5, color=color_list)
        axes[0,0].set_ylabel('50 subtasks \n Success Rate (%)')
        axes[0,0].set_ylim(0, 100)
        axes[0,0].set_xticks(range(15))
        axes[0,0].set_xlim(-0.5,14.5)
        axes[0,0].set_title('Task classifier K: {:d}, seed: {:d}, {:d} subtasks, {:d}M steps \n train: {:.1f}%, test {:.1f}%'.format(
            class_num, seed_list[seed],parametric_num, int((frame_list[iter]+1)/1e6), 100*train_success, 100*test_success))

        for i in range(1,8):
            heatmap_data = np.zeros((class_num, 15), dtype=int)
            heatmap_data = data_[:,:, step_list[i]//max_episode_steps, step_list[i]%max_episode_steps,:]
            heatmap_data = np.mean(heatmap_data, axis=1)
            sns.heatmap(np.flip(np.transpose(heatmap_data),0), ax=axes[i//2,i%2], vmin=0, vmax=1.0, annot=True, fmt='.2f', cmap = 'viridis', cbar=False, xticklabels=range(0, 15),yticklabels=range(class_num, 0, -1))
            axes[i//2,i%2].set_ylabel('Mean_subtasks \n y')
            if i ==0:
                axes[i//2,i%2].set_title("Train Success {:.1f}%, test {:.1f}% \n Trained steps: {:d}M steps, Episode steps: {:d}".format(100*train_success, 100*test_success, int((frame_list[iter]+1)/1e6), step_list[i]))
            else:
                axes[i//2,i%2].set_title("Trained steps: {:d}M steps, Episode steps: {:d}".format(int((frame_list[iter]+1)/1e6), step_list[i]))

        fig.tight_layout(h_pad = 0.2)

        plt.savefig('plots/Episode_probs/episode_probs_{:d}_seed_{:d}M_steps.png'.format(seed_list[seed], int((frame_list[iter]+1)/1e6)))
        plt.show()
        plt.close()










