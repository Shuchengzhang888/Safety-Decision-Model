#Deep Q Network Safety Decision-making Model of Autonomous Vehicle Based on Trajectory Prediction

To overcome the shortcomings of handcrafted decision methods in field of autonomous vehicle, we developed a Safety Decision-making Model based on Deep
Q Network (DQN) to complete safe and high-speed driving in the highway environment. The experiments showed that our model greatly increased average
speed while keeping vehicles safe. Moreover, we added the predicted trajectories of surrounding vehicles into the original input and proved their importance in improving the risk forecast ability.

## Install
Training and testing environmrnt is based on OpenAI/Gym and highway-env packages.
``pip install gym
``pip install highway-env
RL models are based on Baseline3.
``pip install baselines
## Code Description
Folder
code -> all written codes
code/data -> all experiments results
! pip install highway-env
code/_*input_lowhighspeedreward_v* -> model
code/DQN_training.py -> train & test DQN
code/my_highway_env.py -> toy exvironment
code/observation.py -> model input with prediction information
code/kenimatics.py -> predict trajectory and controll
code/env_test.py
code/model_test.py -> test model &env
code/result_record.py -> record experiment results


## Demo
demo -> demo video`
