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

seed_list=[10,11]
listdirs_ml10 = listdirs('logs/logs_ML10Env-v2/')
dir_ml10 = []
for dir in listdirs_ml10:
    if ('varibad_10__29:08_17:19:43' in dir) or ('varibad_11__30:08_09:34:16' in dir):
        dir_ml10.append(dir)
print(dir_ml10)

'''
listdirs_ml10 = listdirs('logs/logs_ML10Env-v2')
seed_list = range(10,14)
dir_ml10=[]
for seed in seed_list:
    for dir in listdirs_ml10:
        if ('varibad_10__22:08_17:51:31' in dir) or ('varibad_11__22:08_17:51:41' in dir) or ('varibad_12__22:08_17:51:52' in dir) or ('varibad_13__22:08_17:52:06' in dir):
            dir_ml10.append(dir)
print(dir_ml10)
'''
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

        data_ = np.load(dir_ml10[seed]+'/'+str(int(iter_list_[iter]))+'/probs.npy')
        parametric_num = np.shape(data_)[1]
        class_num =np.shape(data_)[2]
        taskwise_entropy = np.mean(entropy(data_+1e-20,axis=2),axis=1)
        classified = np.argmax(data_,axis=2)
        mean_softmax = np.mean(data_, axis=1)
        print(classified)

        fig, axes = plt.subplots(4,1, figsize=(8,10), gridspec_kw={'height_ratios': [1, 1,2.0,2.0]})

        #taskwise success
        axes[0].bar(range(1, 16), 100 * task_successes[iter,:,seed], alpha=0.5, color=color_list)
        axes[0].set_ylabel('50 subtasks \n Success Rate (%)')
        axes[0].set_ylim(0, 100)
        axes[0].set_xticks(range(1, 16))
        axes[0].set_xlim(0.5,15.5)
        axes[0].set_title('Task classifier K: {:d}, seed: {:d}, {:d} subtasks, {:d}M steps \n train: {:.1f}%, test {:.1f}%'.format(
            class_num, seed_list[seed],parametric_num, int((frame_list[iter]+1)/1e6), 100*train_success, 100*test_success))

        #axes[1].bar(range(1, 16),taskwise_entropy, yerr=np.std(entropy(data_+1e-20,axis=2),axis=1), alpha=0.5, color=color_list)
        #axes[1].set_ylabel('Mean entropy of y')
        #axes[1].set_xticks(range(1, 16))
        #axes[1].set_xlim(0.5, 15.5)

        axes[1].bar(range(1, 16),np.mean(np.max(data_,axis=2),axis=1), yerr=np.std(np.max(data_,axis=2),axis=1) , alpha=0.5, color=color_list)
        axes[1].set_ylabel('Mean_subtasks \n Max y')
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xticks(range(1, 16))
        axes[1].set_xlim(0.5, 15.5)

        sns.heatmap(np.flip(np.transpose(mean_softmax),0), ax=axes[2], vmin=0, vmax=1.0, annot=True, fmt='.2f', cmap = 'viridis', cbar=False, xticklabels=range(1, 16),yticklabels=range(class_num, 0, -1))
        axes[2].set_ylabel('Mean_subtasks \n y')

        heatmap_data = np.zeros((class_num,15), dtype=int)
        for task in range(15):
            for class_ in range(class_num):
                heatmap_data[class_, task] = np.count_nonzero(classified[task,:]==class_)

        sns.heatmap(np.flip(heatmap_data,0),ax=axes[3], vmin=0, vmax=50, annot=True, fmt='d', cmap = 'viridis', cbar=False, xticklabels=range(1, 16),yticklabels=range(class_num, 0, -1))
        axes[3].set_xlabel('Task Index')
        axes[3].set_ylabel('50 subtasks \n argmax y')



        plt.savefig('plots/Task_classified/Task_classifier_K_{:d}_seed_{:d}_{:d}subtasks_{:d}M_steps.png'.format(class_num, seed_list[seed],parametric_num, int((frame_list[iter]+1)/1e6)))
        plt.show()
        plt.close()












