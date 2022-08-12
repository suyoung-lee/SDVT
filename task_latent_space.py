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

for current_seed in range(8):
    for current_iter in [-1, 4999]:
        train_success = np.mean(task_successes[(current_iter+1)//5000, 0:10, current_seed])
        test_success = np.mean(task_successes[(current_iter+1)//5000, 10:, current_seed])

        x = np.zeros((15*5,5)) #15 tasks, 5 parametric, 5 dimension
        y = np.repeat(range(1,16),5)
        for task in range(15):
            for param in range(5):
                data_ = np.load(dir_ml10[current_seed]+'/'+str(current_iter)+'/'+str(task)+'_'+"{:02d}".format(param)+'_9_data.npz')
                x[5*task+param,:]= data_['episode_latent_means'][-1,:]

        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x)

        palette=sns.color_palette("hls", 10)

        df_train = pd.DataFrame()
        df_train["y"] = y[:50]
        df_train["comp-1"] = z[:50,0]
        df_train["comp-2"] = z[:50,1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df_train.y.tolist(), marker='o',
                        palette=palette,
                        data=df_train).set(title="ML10 T-SNE projection Seed: " + str(current_seed) +', '+ str(int((current_iter+1)//25))+'M steps\n' +
                                           "Train Success {:.2f}%, Test Success {:.2f}%".format(100*train_success, 100*test_success))

        df_test = pd.DataFrame()
        df_test["y"] = y[50:]
        df_test["comp-1"] = z[50:,0]
        df_test["comp-2"] = z[50:,1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df_test.y.tolist(), marker='^',
                        palette=palette[:5],
                        data=df_test).set(title="ML10 T-SNE projection Seed: " + str(current_seed) +', '+ str(int((current_iter+1)//200))+'M steps\n' +
                                           "Train Success {:.2f}%, Test Success {:.2f}%".format(100*train_success, 100*test_success))

        legend_elements = []
        for i in range(10):
            legend_elements.append(Line2D([0], [0], marker='o', color='None', markeredgecolor='w', markerfacecolor=palette[i], label=str(i+1)))
        for i in range(5):
            legend_elements.append(Line2D([0], [0], marker='^', color='None', markeredgecolor='w', markerfacecolor=palette[i], label=str(i+11)))

        plt.subplots_adjust(hspace=0.1, wspace =0.35, bottom=0.35, top=0.90, left=0.15, right=0.97)
        plt.legend(handles = legend_elements, bbox_to_anchor=(0.500,-0.55),loc='lower center', ncol=8, title='task index', handletextpad=0.2, labelspacing=0.2, columnspacing=1.0)
        plt.savefig('plots/Latents/Latents_' + str(current_seed)+'_'+ str(current_iter) + '.png')
        plt.close()