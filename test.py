from gym import envs
import gym
import plotly.graph_objects as go
import pandas as pd
# Create random data with numpy
import numpy as n
X1 = [-1690.832207,
-1690.832207,
-1690.832207,
-1452.99802,
-1452.391763,
-1452.328947,
-1452.328947,
-1452.328947,
-1452.328947,
]

X2 = [-1690.832,
-1690.832,
-1649.376,
-1488.696,
-1492.466,
-1502.593,
-1480.134,
-1486.108,
-1496.24,
]

X3 = [-1690.832,
-1690.832,
-1277.307,
-1302.18,
-1302.147,
-1302.147,
-1302.18,
-1302.147,
-1302.147,
]

Y = [1,
5000,
10000,
15000,
20000,
25000,
30000,
35000,
40000,
]

gloss = pd.read_csv('results_offline_JointGan_softmax_avg_no share_joint_dist.csv')
#dloss = pd.read_csv('d_loss_joint_online.csv')
#reward = pd.read_csv('rewards_vae_online.csv')
fig = go.Figure()
'''
fig.add_trace(go.Scatter(x=reward['time'], y=reward['reward'],
                    mode='lines',
                    name='Reward Collected'))
'''
fig.add_trace(go.Scatter(x=gloss['epoch'], y=gloss['g_loss'],
                    mode='lines',
                    name='Generator Los'))

fig.add_trace(go.Scatter(x=gloss['epoch'], y=gloss['d_loss'],
                    mode='lines', name='Discriminator Loss'))

fig.update_layout(title='JointGAN loss during Offline Training',
                   xaxis_title='Epoch',
                   yaxis_title='Loss')

fig.update_xaxes(ticks="inside")
fig.update_yaxes(ticks="inside")
fig.show()