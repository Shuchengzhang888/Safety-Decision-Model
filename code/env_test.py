import gym
import highway_env
from matplotlib import pyplot as plt


env = gym.make('my-highway-v0')

env.reset()
for _ in range(40):
    obs, reward, done,  info = env.step(env.action_space.sample())
    print(obs)
    env.render()