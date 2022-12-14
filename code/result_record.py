from stable_baselines3 import DQN
import gym
import highway_env
import numpy as np
###Baseline IDM
env = gym.make("my-highway-v0")
model = DQN.load("highway_dqn/_36input_lowhighspeedreward_v1", env=env,device='cpu')

all_velo = []
all_dis = []
collision_times = 0
for i in range(500):
    done = False
    obs= env.reset()

    ini_place = env.unwrapped.vehicle.position[0]
    v = []

    while not done:
      action, _states = model.predict(obs, deterministic=True)
      action = int(action)
      obs, reward, done, info = env.step(action)
      env.render()

      v.append(info['speed'])
      if done is True:
          all_dis.append(env.unwrapped.vehicle.position[0] - ini_place)
          all_velo.append(np.mean(v))
          if info['crashed']:
              collision_times +=1
          print(all_dis[-1], all_velo[-1], collision_times)


import pandas as pd
dict = {'dis': all_dis,'velo': all_velo, 'c': collision_times}
df = pd.DataFrame(dict)
df.to_csv('36input3.csv')
print(np.mean(all_dis), np.mean(all_velo))
