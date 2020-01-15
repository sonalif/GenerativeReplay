from gym import envs
import gym
import joblib
import numpy as np
import torch
from utils import JointGANTrainer
from pygit2 import Repository


env = gym.make('Pendulum-v0')
db1 = joblib.load('replay.joblib')
db2 = joblib.load('replay.joblib')

exp = np.concatenate((db1, db2), 0)

replay = JointGANTrainer(env.observation_space.shape[0], env.action_space.shape[0], 256, 5, env.action_space.low[0],
                         env.action_space.high[0], env.observation_space.low, env.observation_space.high)
epochs = 10
batch_size = 256
print(env.action_space.low)
for epoch in range(epochs):
    id = 0
    #shuffle
    for i in range(int(exp.shape[0]/batch_size)):
        data = exp[id: id+batch_size, 0:4]
        data = replay.normalise(torch.Tensor(data))

        d_loss, g_loss, gen_xy, gen_yx, critic, opt_g, opt_d = replay.train(data)

        print(
            "[Epoch %d|%d] [Batch %d|%d] [D loss: %f] [G loss: %f]"
            % (epoch+1, epochs, i, exp.shape[0]/batch_size, d_loss, g_loss)
        )
        id = id + batch_size

    torch.save(gen_xy, f"./models/GR/offline_{Repository('.').head.shorthand}_{'G_XY'}.pth")

    torch.save(gen_yx,f"./models/GR/offline_{Repository('.').head.shorthand}_{'G_YX'}.pth")
    torch.save(opt_g, f"./models/GR/offline_{Repository('.').head.shorthand}_{'G'}_optimizer.pth")

    torch.save(critic, f"./models/GR/offline_{Repository('.').head.shorthand}_{'D'}_.pth")
    torch.save(opt_d, f"./models/GR/offline_{Repository('.').head.shorthand}_{'D'}_optimizer.pth")
