import matplotlib.pyplot as plt
import numpy as np
import re
import os
import matplotlib as mpl
import pandas as pd
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

train_success = []
train_return = []
test_success = []
test_return = []
steps_per_iter = 2500000
for dir in dir_ml10:
    df = pd.read_csv(dir+'/log_eval.csv')
    train_success.append(df['Mean train success'].values.astype(np.float))
    train_return.append(df['Mean train return'].values.astype(np.float))
    test_success.append(df['Mean test success'].values.astype(np.float))
    test_return.append(df['Mean test return'].values.astype(np.float))

'''
for seed in range(len(train_success)):
    length = len(train_success[seed])
    plt.plot(np.linspace(0, steps_per_iter*(length-1), length),  train_success[seed])
plt.show()
plt.close()
'''

length = len(train_success[0])
train_mean = 100*np.mean(train_success,axis=0)
train_std = 100*np.std(train_success,axis=0)
plt.plot(np.linspace(0, steps_per_iter*(length-1), length), train_mean)
plt.fill_between(np.linspace(0, steps_per_iter*(length-1), length), train_mean-train_std, train_mean+train_std , alpha=0.1)

test_mean = 100*np.mean(test_success,axis=0)
test_std = 100*np.std(test_success,axis=0)
plt.plot(np.linspace(0, steps_per_iter*(length-1), length), test_mean)
plt.fill_between(np.linspace(0, steps_per_iter*(length-1), length), test_mean-test_std, test_mean+test_std , alpha=0.1)

plt.title('ML10 Success Rate (8 seeds)')
plt.xlabel('Steps')
plt.ylabel('Success Rate (%)')
plt.legend(['VariBAD Train', 'VariBAD Test'], loc='upper left')
plt.savefig('plots/Success_Rate_' + '0' + '.png')
plt.show()
plt.close()


train_mean = np.mean(train_return,axis=0)
train_std = np.std(train_return,axis=0)
plt.plot(np.linspace(0, steps_per_iter*(length-1), length), train_mean)
plt.fill_between(np.linspace(0, steps_per_iter*(length-1), length), train_mean-train_std, train_mean+train_std , alpha=0.1)

test_mean = np.mean(test_return,axis=0)
test_std = np.std(test_return,axis=0)
plt.plot(np.linspace(0, steps_per_iter*(length-1), length), test_mean)
plt.fill_between(np.linspace(0, steps_per_iter*(length-1), length), test_mean-test_std, test_mean+test_std , alpha=0.1)

plt.title('ML10 Return (8 seeds)')
plt.xlabel('Steps')
plt.ylabel('Return')
plt.legend(['VariBAD Train', 'VariBAD Test'], loc='upper left')
plt.savefig('plots/Return_' + '0' + '.png')
plt.show()
plt.close()





