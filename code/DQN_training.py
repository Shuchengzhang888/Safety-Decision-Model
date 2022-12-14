import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env
import pandas as pd
import numpy as np

TRAIN = True

if __name__ == '__main__':
    # Create the environment
    #env = gym.make("highway-v0")
    n_cpu = 6
    #batch_size = 256
    env = make_vec_env("my-highway-v0", n_envs=n_cpu)
    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=254,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                device='cpu',
                tensorboard_log="highway_dqn/")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e5))
        model.save("highway_dqn/_44input_lowhighspeedreward_v2")
        del model


##Training with test

# if __name__ == '__main__':
#     for epoch in [5e3,1e4,2e4,3e4,4e4,5e4,6e4,7e4,8e4,9e4,1e5]:
#         n_cpu = 6
#         env = make_vec_env("my-highway-v0", n_envs=n_cpu)
#         model = DQN('MlpPolicy', env,
#                     policy_kwargs=dict(net_arch=[256, 256]),
#                     learning_rate=5e-4,
#                     buffer_size=15000,
#                     learning_starts=200,
#                     batch_size=256,
#                     gamma=0.8,
#                     train_freq=1,
#                     gradient_steps=1,
#                     target_update_interval=50,
#                     verbose=1,
#                     device='cpu',
#                     tensorboard_log="highway_dqn/")

#         if TRAIN:
#             model.learn(total_timesteps=int(epoch))
            
#             env = gym.make("highway-v0")
#             all_velo = []
#             all_dis = []
#             all_reward = []
#             collision_times = 0
#             for i in range(500):
#                 done = False
#                 obs= env.reset()

#                 ini_place = env.unwrapped.vehicle.position[0]
#                 v = []
#                 rew = []
#                 while not done:
#                     action, _states = model.predict(obs, deterministic=True)
#                     action = int(action)
#                     obs, reward, done, info = env.step(action)
#                     #env.render()
#                 v.append(info['speed'])
#                 rew.append(reward)

#                 if done is True:
#                     all_dis.append(env.unwrapped.vehicle.position[0] - ini_place)
#                     all_velo.append(np.mean(v))
#                     all_reward.append(np.mean(rew))
#                     if info['crashed']:
#                         collision_times +=1
#                     print(all_dis[-1], all_velo[-1], collision_times)

#             dic = {'dis': all_dis,'velo': all_velo, 'reward': all_reward, 'c': collision_times,}
#             df = pd.DataFrame(dic)
#             df.to_csv('20input'+str(epoch)+'.csv')
#             print(np.mean(all_dis), np.mean(all_velo))
#             del model