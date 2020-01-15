from gym import envs
import gym
import joblib
import numpy as np
import torch
from utils import JointGANTrainer
from pygit2 import Repository
import pandas as pd

env = gym.make('Pendulum-v0')
db1 = joblib.load('replay.joblib')
db2 = joblib.load('replay.joblib')

exp = np.concatenate((db1, db2), 0)

print(exp.shape)
results = []
epochs = 10
batch_size = 256

replay = JointGANTrainer(env.observation_space.shape[0], env.action_space.shape[0], batch_size, 5, env.action_space.low[0],
                         env.action_space.high[0], env.observation_space.low, env.observation_space.high)

print(env.action_space.low)
count = 0
try:
    for epoch in range(epochs):
        id = 0

        #shuffle
        for i in range(int(exp.shape[0]/batch_size)):
            count = count + 1
            data = exp[id: id+batch_size, 0:4]
            data = replay.normalise(torch.Tensor(data))

            d_loss, g_loss, gen_xy, gen_yx, critic, opt_gxy, opt_gyx, opt_d = replay.train(data)

            print(
                "[Epoch %d|%d] [Batch %d|%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, epochs, i, exp.shape[0]/batch_size, d_loss, g_loss)
            )
            temp = [count, epoch+1, i+1, d_loss, g_loss]

            id = id + batch_size
            results.append(temp)

        torch.save(gen_xy,
                   f"./models/GR/{Repository('.').head.shorthand}_{'G_XY'}_{'Pendulum'}.pth")

        torch.save(opt_gxy,
                   f"./models/GR/{Repository('.').head.shorthand}_{'G_XY'}_{'Pendulum'}_optimizer.pth")

        torch.save(gen_yx,
                   f"./models/GR/{Repository('.').head.shorthand}_{'G_YX'}_{'Pendulum'}.pth")
        torch.save(opt_gyx,
                   f"./models/GR/{Repository('.').head.shorthand}_{'G_YX'}_{'Pendulum'}_optimizer.pth")

        torch.save(critic,
                   f"./models/GR/{Repository('.').head.shorthand}_{'D'}_{'Pendulum'}.pth")
        torch.save(opt_d,
                   f"./models/GR/{Repository('.').head.shorthand}_{'D'}_{'Pendulum'}_optimizer.pth")

except KeyboardInterrupt:
    print('Stopping...')
finally:
    pd.DataFrame(results).to_csv("results_offline_JointGan_softmax_avg_no share_joint_dist.csv")