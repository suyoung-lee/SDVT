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
seed_list = range(10,12)
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
task_returns = np.zeros((len(iter_list), 15,len(seed_list))) #first 10 rows are train
for seed in range(len(seed_list)):
    dir = dir_ml10[seed]
    df = pd.read_csv(dir+'/log_eval.csv')
    keys = df.keys()
    iter_num = 0
    for iter in iter_list:
        for task in range(15):
            task_successes[iter_num, task, seed] = df[keys[17+task]][iter]
            task_returns[iter_num, task, seed] = df[keys[2+task]][iter]
        iter_num+=1

frame_list = np.array(df['frames'][:min_iter].values.astype(np.int))
print(task_successes)

train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
plt.plot(frame_list, train_mean)
plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1)

test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
plt.plot(frame_list, test_mean)
plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1)

plt.title('ML10 Success Rate ({:d} seeds)'.format(len(dir_ml10)))
plt.xlabel('Steps')
plt.ylabel('Success Rate (%)')
plt.legend(['VariBAD Train', 'VariBAD Test'], loc='upper left')
plt.savefig('plots/Success_Rate_seed_{:d}to{:d}.png'.format(seed_list[0],seed_list[-1]))
plt.show()
plt.close()


train_mean = 100*np.mean(np.mean(task_returns[:,:10,:], axis=1),axis=1)
train_std = 100*np.std(np.mean(task_returns[:,:10,:], axis=1),axis=1)
plt.plot(frame_list, train_mean)
plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1)

test_mean = 100*np.mean(np.mean(task_returns[:,10:,:], axis=1),axis=1)
test_std = 100*np.std(np.mean(task_returns[:,10:,:], axis=1),axis=1)
plt.plot(frame_list, test_mean)
plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1)

plt.title('ML10 Return ({:d} seeds)'.format(len(dir_ml10)))
plt.xlabel('Steps')
plt.ylabel('Return')
plt.legend(['VariBAD Train', 'VariBAD Test'], loc='upper left')
plt.savefig('plots/Return_seed_{:d}to{:d}.png'.format(seed_list[0],seed_list[-1]))
plt.show()
plt.close()




