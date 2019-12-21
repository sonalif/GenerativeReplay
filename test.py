from gym import envs
import gym
env = gym.make('Assault-v0')
print(env.action_space)
print(env.observation_space.high.shape)
print(env.observation_space.low)
print(type(env.observation_space))
input()
for i_episode in range(20):
    observation = env.reset()
    print(len(observation))
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        #print(action)
        observation, reward, done, info = env.step(action)
        #print(reward)
        #print(observation)
        #print(info)
        #print('-' * 100)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()