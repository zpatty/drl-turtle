import os
import sys
import json
import time
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

import gymnasium as gym
from EPHE import EPHE
from AukeCPG import AukeCPG

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_rl/turtle_rl"
sys.path.append(submodule)

ENV_NAME = 'HalfCheetah-v4'
PARAM_FILE = 'best_params.pth'
POLICY_FILE = 'policy.pkl'

def set_random_seed(seed):
    np.random.seed(seed)

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

def train(num_params=20, num_mods=10):
    """
    Implements EPHE algorithm, where at each episode we sample params from N(v|h) M times.
    We then choose the best K params that are selected from the sorted reward of R(v^m).
    
    """
    trial_folder = input("choose folder name: ")
    best_param_fname = trial_folder + f'/best_params_ephe.pth'

    env = gym.make(ENV_NAME, render_mode='human')
    # env = gym.make(ENV_NAME)
    mu = np.array([5.0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        
    # mu = np.random.rand((num_params))
    # mu[9] = 0.1
    # mu[10] = 0.1
    # mu[0] = 5.0

    sigma = np.ones(num_params) * 0.7
    sigma[0] = 0.1
    params, config_params = parse_learning_params()

    a_r  = params["a_r"]
    a_x = params["a_x"]
    phi = params["phi"]
    w = params["w"]
    M = params["M"]
    K = params["K"]
    dt = params["dt"]

    cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)

    ephe = EPHE(
        
        # We are looking for solutions whose lengths are equal
        # to the number of parameters required by the policy:
        solution_length=mu.shape[0],
        
        # Population size: the number of trajectories we run with given mu and sigma 
        popsize=M,
        
        # Initial mean of the search distribution:
        center_init=mu,
        
        # Initial standard deviation of the search distribution:
        stdev_init=sigma,

        # dtype is expected as float32 when using the policy objects
        dtype='float32', 

        K=K
    )

    # Bo Chen paper says robot converged in 20 episodes
    max_episodes = 10
    max_episode_length = params["rollout_time"]/dt

    # data structs for plotting
    param_data = np.zeros((num_params, M, max_episodes))
    mu_data = np.zeros((num_params, max_episodes))
    sigma_data = np.zeros((num_params, max_episodes))
    reward_data = np.zeros((M, max_episodes))
    os.makedirs(trial_folder, exist_ok=True)

    best_params = np.zeros((num_params))
    best_reward = 0

    # Specify the file path where you want to save the JSON file
    file_path = trial_folder + "/config.json"

    for episode in range(max_episodes):
        print(f"episode: {episode}")
        # The main loop of the evolutionary computation
        # this is where we run our M trajectories
        lst_params = np.zeros((num_params, M))
        # Get the M solutions from the ephe solver
        solutions = ephe.ask()          
        R = np.zeros(M)
        folder_name = trial_folder + f"/CPG_episode_{episode}"
        for i in range(M):
            subplot=True
            lst_params[:, i] = solutions[i]
            fitness, total_actions = cpg.set_params_and_run(env=env, policy_parameters=solutions[i], max_episode_length=max_episode_length)
            print(f"raw reward: {fitness}")
            timesteps = total_actions.shape[1] - 1
            t = np.arange(0, timesteps*dt, dt)
            if fitness > best_reward:
                print(f"params with best fitness: {solutions[i]} with reward {fitness}\n")
                best_reward = fitness
                best_params = solutions[i]
            if subplot:
                # Plotting each row as its own subplot
                fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
                for j, ax in enumerate(axs):
                    ax.plot(t, total_actions[j, 1:])
                    ax.set_title(f"CPG {j+1}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Data")
                    ax.grid(True)
                plt.tight_layout()
                # plt.show()
                os.makedirs(folder_name, exist_ok=True)
                plt.savefig(folder_name + f'/CPG_output_run{i}_reward_{fitness}.png')
            if fitness < 0:
                R[i] = 0
            else:
                R[i] = fitness
        print("--------------------- Episode:", episode, "  median score:", np.median(R), "------------------")
        print(f"all rewards: {R}\n")
        # get indices of K best rewards
        best_inds = np.argsort(-R)[:K]
        k_params = lst_params[:, best_inds]
        print(f"k params: {k_params}")
        k_rewards = R[best_inds]
        print(f"k rewards: {k_rewards}")
        # We inform our ephe solver of the fitnesses we received,
        # so that the population gets updated accordingly.
        ephe.update(k_rewards=k_rewards, k_params=k_params)
        print(f"new mu: {ephe.center()}\n")
        print(f"new sigma: {ephe.sigma()}\n")
        # save param data
        param_data[:, :, episode] = lst_params
        # save mus and sigmas 0.00639871]
        mu_data[:, episode] = ephe.center()
        sigma_data[:, episode] = ephe.sigma()
        reward_data[:, episode] = R
    best_mu = ephe.center()
    best_sigma = ephe.sigma()
    print(f"best mu: {best_mu}\n")
    print(f"best sigma: {best_sigma}\n")
    print(f"best params: {best_params} got reward of {best_reward}\n")
    # save the best params
    torch.save(best_params, best_param_fname)
    # save data structs to matlab 
    scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data, 'param_data': param_data, 'reward_data': reward_data})
    print(f"data saved in folder named: {trial_folder}")
    return best_params

######################################################################################################################
def test(best_params):

    # instantiate gym environment 
    env = gym.make(ENV_NAME, render_mode="human", camera_id=1)
    print("made env")
    # load parameters of final solution into the policy
    params, config_params = parse_learning_params()

    a_r  = params["a_r"]
    a_x = params["a_x"]
    phi = params["phi"]
    w = params["w"]
    M = params["M"]
    K = params["K"]
    dt = params["dt"]

    cpg = AukeCPG(num_params=13, num_mods=6, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)

    cpg.set_parameters(best_params)
    print("set params")
    # Now we test out final policy
    cumulative_reward = 0.0

    # Reset the environment
    print("attempt reset")
    print(env.reset())

    # Main loop of the trajectory
    while True:

        action = cpg.get_CPG_output()
        observation, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        env.render()
        # env._get_viewer('human').render(camera_id=0) 
        cumulative_reward += reward

        if done:
            break
    
    return cumulative_reward


def main(args=None):
    best_params = train(num_params=13, num_mods=6)
    # best_params = torch.load('test/best_params_ephe.pth')
    print(f"best params: {best_params}")
    test_reward = test(best_params=best_params)
    print(f"test reward: {test_reward}")
if __name__ == '__main__':
    main()
