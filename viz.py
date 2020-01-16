import numpy as np
import pandas as pd

with open('ii.txt', 'r') as f:
    doc = f.readlines()


g_loss = []
d_loss = []
reward = []
first = 0
avg_reward = []
count = 0

t = 0
for i, line in enumerate(doc):

    count = count+1
    line = line.replace('\n', '')
    if line.startswith('Total'):
        t = t + 200
        reward.append([t, float(line[-9:].replace(" ", ''))])
        print(t)

    if line.startswith('[Epoch'):
        print('hskjadn')
        if i == 0:
            g_loss.append([101, float(line[line.index('G')+8: line.index('G') + 16])])
            d_loss.append([101, float(line[line.index('D')+8: line.index('D') + 16])])
        if first == 0:
            first = 1
            g_loss.append([t+1, float(line[line.index('G')+8: line.index('G') + 16])])
            d_loss.append([t+1, float(line[line.index('D')+8: line.index('D') + 16])])
        if first == 1:
            first = 0
            g_loss.append([t + 101, float(line[line.index('G')+8: line.index('G') + 16])])
            d_loss.append([t + 101, float(line[line.index('D')+8: line.index('D') + 16])])
    if line.startswith('Evaluation '):
        avg_reward.append([count, float(line[-9:].replace(" ", ""))])

pd.DataFrame(avg_reward).to_csv("avg_reward_joint_online.csv")
pd.DataFrame(reward).to_csv("rewards_joint_online.csv")
pd.DataFrame(g_loss).to_csv("g_loss_joint_online.csv")
pd.DataFrame(d_loss).to_csv("d_loss_joint_online.csv")