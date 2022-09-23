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


plt.figure(figsize=(10,8))


toggle = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
toggle = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1]
#toggle = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]



palette=sns.color_palette("hls", 8)
colornum=0

if toggle[0]:
    listdirs_ml10 = listdirs('ml10-joint-log/VB_vanilla/logs_ML10Env-v2/')
    seed_list = range(10,14)
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if "varibad_"+str(seed)+'_' in dir:
                dir_ml10.append(dir)
    print(dir_ml10)

    train_success = []
    train_return = []
    test_success = []
    test_return = []

    iter_list = []
    for dir in dir_ml10:
        df = pd.read_csv(dir+'/log_eval.csv')
        iter_list.append(len(df))
    min_iter = min(iter_list)

    for dir in dir_ml10:
        df = pd.read_csv(dir+'/log_eval.csv')
        train_success.append(df['Mean train success'][:min_iter].values.astype(np.float))
        train_return.append(df['Mean train return'][:min_iter].values.astype(np.float))
        test_success.append(df['Mean test success'][:min_iter].values.astype(np.float))
        test_return.append(df['Mean test return'][:min_iter].values.astype(np.float))

    step_list = np.array(df['frames'][:min_iter].values.astype(np.int))

    length = len(train_success[0])
    train_mean = 100*np.mean(train_success,axis=0)
    train_std = 100*np.std(train_success,axis=0)
    plt.plot(step_list, train_mean, color=palette[colornum], label='VB')
    plt.fill_between(step_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(test_success,axis=0)
    test_std = 100*np.std(test_success,axis=0)
    plt.plot(step_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(step_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1





if toggle[1]:
    listdirs_ml10 = listdirs('ml10-v3/logs/logs_ML10Env-v2')
    seed_list = range(10,14)
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-DISC')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1


if toggle[2]:
    listdirs_ml10 = listdirs('ml10-v4/logs/logs_ML10Env-v2')
    seed_list = range(10,14)
    date = '12:08'
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if "varibad_"+str(seed)+'__'+date in dir:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1


    plt.tight_layout(pad=2.00)




if toggle[3]:
    listdirs_ml10 = listdirs('ml10-joint-log/logs_ML10Env-v2/vae_10x/logs_ML10Env-v2')
    seed_list = [10,12,13]  #seed 11 is broken, seed 12 is the best
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    print('='*50, np.mean(task_successes[-1,:10,:], axis=0))
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB FT')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1



if toggle[4]:
    listdirs_ml10 = listdirs('ml10-joint-log/logs_ML10Env-v2')
    seed_list = range(10,14)
    date = '12:08'
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if ('varibad_10__17:08_17:53:57' in dir) or ('varibad_11__17:08_17:53:58' in dir) or ('varibad_12__17:08_17:54:03' in dir) or ('varibad_12__17:08_17:54:03' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix 10xCatloss')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1


if toggle[5]:
    listdirs_ml10 = listdirs('ml10-joint-log/logs_ML10Env-v2')
    seed_list = range(10,14)
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if ('varibad_10__17:08_17:54:09' in dir) or ('varibad_11__17:08_17:54:12' in dir) or ('varibad_12__17:08_17:54:15' in dir) or ('varibad_13__17:08_17:54:17' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix K=20')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1







if toggle[6]:
    listdirs_ml10 = listdirs('ml10-v5/logs/logs_ML10Env-v2')
    seed_list = range(10,14)
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if ('varibad_10__22:08_17:26:22' in dir) or ('varibad_11__22:08_17:26:54' in dir) or ('varibad_12__22:08_17:28:14' in dir) or ('varibad_12__22:08_17:28:14' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix K=5')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1



if toggle[7]:
    listdirs_ml10 = listdirs('ml10-v5/logs/logs_ML10Env-v2')
    seed_list = range(10,14)
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if ('varibad_10__22:08_17:51:31' in dir) or ('varibad_11__22:08_17:51:41' in dir) or ('varibad_12__22:08_17:51:52' in dir) or ('varibad_13__22:08_17:52:06' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix 2xCatloss')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1




if toggle[8]:
    listdirs_ml10 = listdirs('ml10-joint-log/logs_ML10Env-v2/vae_10x_GM/logs_ML10Env-v2/')
    seed_list = range(10,14)
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if ('varibad_10__25:08_13:46:20' in dir) or ('varibad_11__25:08_13:46:28' in dir) or ('varibad_12__25:08_13:46:31' in dir) or ('varibad_13__25:08_13:46:35' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix, FT, K=5, d=10')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

if toggle[9]:
    listdirs_ml10 = listdirs('ml10-joint-log/logs_ML10Env-v2/vae_10x_GM/logs_ML10Env-v2/')
    seed_list = range(10,14)
    dir_ml10=[]
    for seed in seed_list:
        for dir in listdirs_ml10:
            if ('varibad_10__25:08_13:47:14' in dir) or ('varibad_11__25:08_13:47:18' in dir) or ('varibad_12__25:08_13:47:21' in dir) or ('varibad_13__25:08_13:47:23' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=10, d=5')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)


if toggle[10]:
    listdirs_ml10 = listdirs('ml10-v5/logs/logs_ML10Env-v2/')
    seed_list = range(10, 14)
    dir_ml10=[]
    for dir in listdirs_ml10:
        if ('varibad_10__29:08_14:44:09' in dir) or ('varibad_11__29:08_14:48:15' in dir) or ('varibad_12__29:08_14:48:31' in dir) or ('varibad_13__29:08_14:48:45' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB FT, d=20')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)


plotnum = 11
if toggle[11]:
    listdirs_ml10 = listdirs('ml10-v5/logs/logs_ML10Env-v2/')
    seed_list = range(10, 13)
    dir_ml10=[]
    for dir in listdirs_ml10:
        if ('varibad_10__29:08_17:19:43' in dir) or ('varibad_11__30:08_09:34:16' in dir) or ('varibad_12__31:08_09:04:24' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=10, d=5, avg elbo')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)

plotnum = 12
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-v5/logs/vae10x_4layerDecR/logs_ML10Env-v2/')
    seed_list = [10, 11, 12, 13]

    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB FT 4-layer Rdec')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1
    plt.tight_layout(pad=2.00)

plotnum = 13
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/vae10x_alphaR0.1/logs_ML10Env-v2/')
    seed_list = [10, 11, 12, 13]

    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB FT, alphaR=0.1')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1
    plt.tight_layout(pad=2.00)


plotnum = 14
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/vae10x_c5_d5_alphaR0.01/logs_ML10Env-v2/')
    seed_list = range(10, 14)
    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=5, d=5, alphaR=0.01')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)

plotnum = 15
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/vae10x_c10_d20_alphaR0.01/logs_ML10Env-v2/')
    seed_list = [11, 12, 13]

    dir_ml10=[]
    for dir in listdirs_ml10:
        if ('varibad_11__31:08_09:41:50' in dir) or ('varibad_12__30:08_16:26:21' in dir) or ('varibad_13__30:08_16:26:27' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=10, d=20, alphaR=0.01')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)

plotnum = 16
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-v6/logs/vae10x_c10_d10_alphaR0.01_subsampleGM/logs_ML10Env-v2/')
    seed_list = [10, 11, 12]

    dir_ml10=[]
    for dir in listdirs_ml10:
        if ('varibad_10__06:09_14:58:19' in dir) or ('varibad_11__07:09_09:07:28' in dir) or ('varibad_12__07:09_09:07:49' in dir):
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=5, d=5, avgELBO, subsample')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)


plotnum = 17
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/vae10x_c5_d5_passprob_avgvae_adjustalpha/logs_ML10Env-v2/')
    seed_list = [10, 11, 12, 13]

    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=5, d=5, avgELBO, dropout0.5')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)



plotnum = 18
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/vae10x_c10_d10_avgvae_adjustalpha_lrvae1e-4/logs_ML10Env-v2/')
    seed_list = [10, 11, 12, 13]

    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=10, d=10, avgELBO, lrvae 1e-4')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)

plotnum = 19
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/vae10x_c5_d5_passprob_avgvae_adjustalpha_resample0.5/logs_ML10Env-v2/')
    seed_list = [10, 11, 12, 13]

    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=5, d=5, avgELBO, resample')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)


plotnum = 20
if toggle[plotnum]:
    listdirs_ml10 = listdirs('ml10-joint-log/oracle_vae10x_c5_d5_passprob_avgvae_adjustalpha/logs_ML10Env-v2/')
    seed_list = [10, 11]

    dir_ml10=[]
    for dir in listdirs_ml10:
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
    print(frame_list)
    print(task_successes)

    train_mean = 100*np.mean(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    train_std = 100*np.std(np.mean(task_successes[:,:10,:], axis=1),axis=1)
    plt.plot(frame_list, train_mean, color=palette[colornum], label='VB-mix FT, K=5, d=5, avgELBO, oracle')
    plt.fill_between(frame_list, train_mean-train_std, train_mean+train_std , alpha=0.1, color=palette[colornum])

    test_mean = 100*np.mean(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    test_std = 100*np.std(np.mean(task_successes[:,10:,:], axis=1),axis=1)
    plt.plot(frame_list, test_mean, color=palette[colornum], linestyle='--')
    plt.fill_between(frame_list, test_mean-test_std, test_mean+test_std , alpha=0.1, color=palette[colornum])
    colornum+=1

    plt.tight_layout(pad=2.00)


plt.title('ML10 Success Rate')
plt.xlabel('Steps')
plt.ylabel('Success Rate (%)')
plt.legend( loc='upper left', ncol=2)
plt.savefig('ml10_plots/Success_Rate2.png')
plt.show()
plt.close()

