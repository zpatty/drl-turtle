import os
import sys
import json
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from AukeCPG import AukeCPG
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from CPG_gym import *
import numpy as np
import pickle

import gymnasium as gym
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_rl/turtle_rl"
sys.path.append(submodule)

ENV_NAME = 'HalfCheetah-v4'
        
def load_params(fname):
    best_params = torch.load(fname)
    return best_params

def save_policy(policy, fname):
    with open(fname, 'wb') as f:
        pickle.dump(policy, f)

def load_policy(fname):
    with open(fname, 'rb') as f:
        policy = pickle.load(f)
    return policy

def save_run(folder_name, i):
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(folder_name + f'CPG_output_run{i}.png')

def parse_learning_params():
    with open('rl_config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    
    return param, config_params

def train(num_params=20, num_mods=10, M=20, K=3):
    trial_folder = input("give folder name: ")
    best_param_fname = trial_folder + f'/best_params.pth'
    params, config_params = parse_learning_params()

    a_r  = params["a_r"]
    a_x = params["a_x"]
    phi = params["phi"]
    w = params["w"]
    M = params["M"]
    K = params["K"]
    dt = params["dt"]
    max_episode_length = 20/0.05

    cheetah = gym.make(ENV_NAME)
    cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)
    env = CPGGym(cheetah, cpg, max_episode_length)
    policy = 'MlpPolicy'
    # policy = SACPolicy(observation_space=env.observation_space,
    #                    action_space=env.action_space,
    #                     lr_schedule=0.001,
    #                      net_arch=[64, 64, 64] )
    # TODO: notes for custom policy
        # what exactly is our observation space(i.e what needs to be passed in for SAC to learn?)
        # how do we want to parameterize SAC
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1
    )
    model.learn(total_timesteps=8/0.05, log_interval=10)
    model.save("half_cheetah_sac")

######################################################################################################################
# def test():

#     # instantiate gym environment 
#     env = gym.make(ENV_NAME, render_mode="human", camera_id=1)

    
#     model = SAC.load("half_cheetah_sac")

#     obs, info = env.reset()
#     total_rewards = 0
#     max_episode_steps = 800
#     t = 0
#     while True:
#         action, _states = model.predict(obs)
#         # action in this case is the CPG params
#         cpg.set_parameters(policy_parameters)

        
#         observation, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         t += 1
#         total_rewards += reward
#         if t > max_episode_steps:
#             break
    
#     return total_rewards


def main(args=None):
    train(num_params=13, num_mods=6, M=40, K=5)
    # total_rewards = test()
if __name__ == '__main__':
    main()
