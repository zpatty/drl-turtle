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

ENV_NAME = 'Ant-v4'
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
    num_trials = 10
    trial_folder = input("choose folder name: ")
    best_param_fname = trial_folder + f'/best_params_ephe.pth'

    # env = gym.make(ENV_NAME, render_mode='human', ctrl_cost_weight=0.2)
    env = gym.make(ENV_NAME, ctrl_cost_weight=0.2)
    params, config_params = parse_learning_params()
    mu = np.array(params["mu"])
    # sigma = np.array(params["sigma"])
    sigma = np.ones(num_params) * 0.7
    sigma[0] = 1
    print(f"starting mu: {mu}")
    print(f"starting sigma: {sigma}")
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
    print(f"max episode length: {max_episode_length}\n")

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
            fitness, total_actions = cpg.set_params_and_run(env=env, policy_parameters=solutions[i], max_episode_length=max_episode_length, PD=True)
            print(f"raw reward: {fitness}")
            timesteps = total_actions.shape[1] - 1
            t = np.arange(0, timesteps*dt, dt)
            if fitness > best_reward:
                print(f"params with best fitness: {solutions[i]} with reward {fitness}\n")
                best_reward = fitness
                best_params = solutions[i]
            if subplot:
                try: 
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
                except:
                    print("probably truncated issue")
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

def test(best_params, num_params=20, num_mods=10):
    num_trials = 10
    # instantiate gym environment 
    env = gym.make(ENV_NAME, render_mode="human", camera_id=1, ctrl_cost_weight=0.2)
    print("made env")
    # load parameters of final solution into the policy
    params, config_params = parse_learning_params()
    cumulative_rewards = []
    a_r  = params["a_r"]
    a_x = params["a_x"]
    phi = params["phi"]
    w = params["w"]
    M = params["M"]
    K = params["K"]
    dt = params["dt"]
    max_episode_length = 60/dt
    PD = True
    cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)

    cpg.set_parameters(best_params)
    print("set params")
    for i in range(num_trials):
        # Now we test out final policy
        cumulative_reward = 0.0

        # Reset the environment
        observation, __ = env.reset()
        # Main loop of the trajectory
        total_actions = np.zeros((num_mods, 1))
        t = 0
        r_forward = 0
        r_ctrl = 0
        while True:

            if PD:
                # pass CPG into PD controller of half cheetah
                qd = cpg.get_CPG_output()
                total_actions = np.append(total_actions, qd.reshape((num_mods,1)), axis=1)

                desired_torques = cpg.kp * (qd - env.data.qpos[-num_mods:])- cpg.kd * env.data.qvel[-num_mods:]
                action = np.clip(desired_torques, -1.0, 1.0)  # clip to action bounds of Half-Cheetah
            else:
                action = cpg.get_CPG_output()

            # print(f"action shape: {action.shape}")
            # total_actions = np.append(total_actions, action.reshape((6,1)), axis=1)
            # print(f"params: {self.params}\n")
            # print(f"action: {action}\n")
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            cumulative_reward += reward
            # print(info)
            # t = time.time()
            # print(f"reward and cost: {(info['reward_forward'], info['reward_ctrl'])}")
            r_forward += info['reward_forward']
            r_ctrl += info['reward_ctrl']
            t += 1
            if t > max_episode_length:
                # print(f"we're donesies with {t}")
                break
            if done:
                if truncated:
                    print("truncccc")
                if terminated:
                    print("terminator")
                break
        print(f"reward forward: {r_forward}\n")
        print(f"reward ctrl: {r_ctrl}\n")
        print(f"cumulative reward: {cumulative_reward}\n")
        cumulative_rewards.append(cumulative_reward)
    return cumulative_rewards


def main(args=None):
    best_params = train(num_params=17, num_mods=8)
    # best_params = torch.load('trial_2_ctrl_cost_0.2/best_params_ephe.pth')
    print(f"best params: {best_params}")
#     best_params = [3.319903,   0.580503,   3.2540421,  0.6300378,  0.81698835, 0.27369037,
#  0.80086046, 0.49979705, 1.2155186,  0.18099527, 0.40686253, 1.1924555,
#  0.38035464, 0.6638087,  0.4573595,  0.25633025, 0.27281272]
    test_rewards = test(best_params=best_params, num_mods=8, num_params=17)
    print(f"test reward: {test_rewards}")


#     k params: [[2.67043662 2.78077984 2.74486589 2.78928089 2.51927018]
#  [1.03507888 0.22093749 0.3441219  0.55798745 0.35031152]
#  [2.24156356 2.11106992 2.32639575 2.16270804 2.32645702]
#  [0.55664909 0.42538255 0.54313141 0.41850206 0.47862184]
#  [0.50355798 0.2910718  0.42689821 0.61190587 0.40184972]
#  [0.6463241  0.73473066 0.7375626  0.56928998 0.6832791 ]
#  [0.95057565 0.56350493 1.39793789 1.32576644 0.9350549 ]
#  [0.46419194 0.57957846 0.68883502 0.61488152 0.72927332]
#  [1.50745285 1.59159768 1.68684256 1.63444984 1.48990154]
#  [0.50805122 0.62058818 0.34312099 0.47532955 0.48858044]
#  [0.52350742 0.46006402 0.53179419 0.53194565 0.51720971]
#  [0.59441346 0.58282644 0.62515831 0.6249969  0.548998  ]
#  [0.19541591 0.12548481 0.18911238 0.04924233 0.14962794]
#  [0.6331777  0.71682131 0.67808598 0.65752953 0.58639508]
#  [8.99602318 8.98680305 8.58455276 8.07146263 8.93037415]
#  [0.28817436 0.2543492  0.23782934 0.20571496 0.2646924 ]
#  [0.3637749  0.40792909 0.42963037 0.44762677 0.38373342]]
# k rewards: [351.20526363 320.0340061  298.20362226 287.68594508 277.07528055]
if __name__ == '__main__':
    main()
