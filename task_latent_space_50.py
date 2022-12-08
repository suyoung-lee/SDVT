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

def listdirs(path):
    list = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sorted(list)


def moving_average(data, M=5):
    temp_data = np.concatenate((np.repeat(data[0], int(M / 2)), data, np.repeat(data[-1], int(M / 2))))
    data = np.convolve(temp_data, np.ones(M) / M, mode='valid')
    return data


listdirs_ml10 = listdirs('logs/logs_ML10Env-v2')
seed_list = [10,11,12,13]
#seed_list = [17]
dir_ml10=[]
for seed in seed_list:
    for dir in listdirs_ml10:
        if "varibad_"+str(seed)+'_' in dir:
            dir_ml10.append(dir)
print(dir_ml10)

#iter_list = [0,1,2,3,4,5,6]
iter_list = [60]
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


for seed in range(len(seed_list)):
    for iter in range(len(iter_list)):
        train_success = np.mean(task_successes[iter, 0:10, seed])
        test_success = np.mean(task_successes[iter, 10:, seed])

        data_ = np.load(dir_ml10[seed]+'/'+str(int(iter_list_[iter]))+'/latent_means.npy')
        param_num = np.shape(data_)[1]
        latent_dim = np.shape(data_)[2]
        x = np.reshape(data_, (15*param_num,latent_dim))
        y = np.repeat(range(1, 16), param_num)

        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x)
        print(z)

        palette=sns.color_palette("hls", 10)

        df_train = pd.DataFrame()
        df_train["y"] = y[:10*param_num]
        df_train["comp-1"] = z[:10*param_num,0]
        df_train["comp-2"] = z[:10*param_num,1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df_train.y.tolist(), marker='o',
                        palette=palette,
                        data=df_train).set(title="ML10 T-SNE projection Seed: " + str(seed_list[seed]) +', '+ str(int((frame_list[iter]+1)/1e6))+'M steps\n' +
                                           "Train Success {:.2f}%, Test Success {:.2f}%".format(100*train_success, 100*test_success))

        df_test = pd.DataFrame()
        df_test["y"] = y[10*param_num:]
        df_test["comp-1"] = z[10*param_num:,0]
        df_test["comp-2"] = z[10*param_num:,1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df_test.y.tolist(), marker='^',
                        palette=palette[1::2],
                        data=df_test).set(title="ML10 T-SNE projection Seed: " + str(seed_list[seed]) +', '+ str(int((frame_list[iter]+1)/1e6))+'M steps\n' +
                                           "Train Success {:.2f}%, Test Success {:.2f}%".format(100*train_success, 100*test_success))

        legend_elements = []
        for i in range(10):
            legend_elements.append(Line2D([0], [0], marker='o', color='None', markeredgecolor='w', markerfacecolor=palette[i], label=str(i+1)))
        for i in range(5):
            legend_elements.append(Line2D([0], [0], marker='^', color='None', markeredgecolor='w', markerfacecolor=palette[2*i-1], label=str(i+11)))

        plt.subplots_adjust(hspace=0.1, wspace =0.35, bottom=0.35, top=0.90, left=0.15, right=0.97)
        plt.legend(handles = legend_elements, bbox_to_anchor=(0.500,-0.55),loc='lower center', ncol=8, title='task index', handletextpad=0.2, labelspacing=0.2, columnspacing=1.0)
        plt.savefig('plots/Latents/Latents_seed_{:d}_frames_{:d}M.png'.format(seed_list[seed], int((frame_list[iter]+1)/1e6),))
        plt.close()