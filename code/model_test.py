from stable_baselines3 import DQN
import gym
import highway_env
import numpy as np

env = gym.make("my-highway-v0")
model = DQN.load("highway_dqn/_36input_lowhighspeedreward_v1", env=env,device='cpu')


while True:
  done = truncated = False
  obs= env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    obs, reward, done, info = env.step(action)
    env.render()