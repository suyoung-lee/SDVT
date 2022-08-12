import numpy as np
import csv
'''
data = np.load('logs/logs_ML10Env-v0/varibad_73__28:07_20:29:00/-1/0_00_0_data.npz')
print(data)

print(np.shape(data['episode_latent_means']))
print(data['episode_latent_means'])
print(np.shape(data['episode_rewards']))
print(np.shape(data['episode_returns']))
print(np.shape(data['episode_successes']))
'''

header = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-close-v2', 'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'sweep-v2', 'basketball-v2']

with open('dummy_file.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

data = [ 151.56644249,  413.80218506, 4125.96897888, 3845.36790466,   15.39050865,
 4325.46099854,   18.29781723,  196.72709656,  158.58665276,  253.13560009]
with open('dummy_file.csv', 'a', encoding='UTF8') as f:
    writer =csv.writer(f)
    writer.writerow(data)
