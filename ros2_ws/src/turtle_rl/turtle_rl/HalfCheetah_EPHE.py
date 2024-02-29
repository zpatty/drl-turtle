import os
import sys
import json
import torch
import time
import scipy
from torch import nn
from copy import copy, deepcopy
import numpy as np
from typing import Optional, Union, Iterable, List, Dict, Tuple, Any
from numbers import Real, Integral
from pgpelib import PGPE
from pgpelib.policies import LinearPolicy, MLPPolicy
from pgpelib.restore import to_torch_module
import matplotlib.pyplot as plt


import numpy as np
import pickle
import torch

import gymnasium as gym
from EPHE import EPHE

ParamVector = Union[List[Real], np.ndarray]
Action = Union[List[Real], np.ndarray, Integral]
# from gym.version import VERSION
# print(f"VERSION: {VERSION}")
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_rl/turtle_rl"
sys.path.append(submodule)

ENV_NAME = 'HalfCheetah-v4'
PARAM_FILE = 'best_params.pth'
POLICY_FILE = 'policy.pkl'

def set_random_seed(seed):
    np.random.seed(seed)

# set random seed
# seed = 1
# set_random_seed(seed=seed)
# print(f"seed: {seed}\n")

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

def positive_int_or_none(x) -> Union[int, None]:
    if x is None:
        return None
    x = int(x)
    if x <= 0:
        x = None
    return x




class DualCPG:
    """
    Bo Chen implementation (the dual-neuron model)
    paper references:
        : https://arxiv.org/pdf/2307.08178.pdf
    """
    def __init__(self, 
                 num_params=21,
                 num_mods=10,
                 B1=5,
                 alpha=10,
                 omega=np.random.rand() * np.pi * 2,
                 observation_normalization=True,
                 fix_B=False,
                 seed=0.0):
        # self._observation_space, self._action_space = (
        #     get_env_spaces(self._env_name, self._env_config)
        # )

        self._seed = seed                                   # seed to replicate 
        self.alpha = alpha                                  # mutual inhibition weight
        self.omega = omega                                  # inter-module connection weight of the neuron
        self.num_mods = num_mods                            # number of CPG modules
        # self.y = np.random.rand(num_mods, 2) * 0.1          # holds the output ith CPG mod (2 neurons per cpg mod)
        self.y = np.zeros((num_mods, 2))

        self.B1 = B1                                        # the first phase is fixed                             
        self.params = np.random.rand((num_params)) * 0.1    # holds the params of the CPG [tau, B, E] = [freq, phase, amplitude]
        # self.U = np.random.rand(num_mods, 2) * 0.1          # the U state of the CPG
        # self.V = np.random.rand(num_mods, 2) * 0.1          # the V state of the CPG
        self.U = np.zeros((num_mods, 2))
        self.V = np.zeros((num_mods, 2))
        self.fix_B = fix_B                                  # whether or not we fix the first B1 like in BoChen
        self.first_time = True

        print(f"starting alpha: {self.alpha}\n")
        print(f"starting omega: {self.omega}\n")
        print(f"number of CPG modules: {self.num_mods}\n")
        if self.fix_B:
            print(f"fixed B1: {self.B1}\n")
    def set_parameters(self, params):
        """
        # TODO: for some reason BoChen only learns B2-->Bn 
        # it isn't clear in paper how they learn/or choose B1
        Updates parameters of the CPG oscillators.
        We currently have the structure to be a 20x1 vector like so:
        = tau: frequency for all oscillators
        = B1 : phase offset for CPG mod 2
        = ...        = ...

        = Bn: phase offset for CPG mod n
        = E1 : amplitude for CPG mod 1
        = ...
        = En: amplitude for CPG mod n
        """
        # make sure phase is kept between 0 and 2pi
        scaled = params.copy()
        # print(f"scale: {type(scaled)}")
        # scale the phase
        # scaled[1:self.num_mods + 1]= (scaled[1:self.num_mods + 1]) % (2*np.pi)
        self.params = scaled
        # print(f"current params: {self.params}")
    
    def get_params(self):
        return self.params
    def set_weights(self, weights):
        """
        Change the intrinsic weights of the CPG modules. 
        alpha is the mutual inhibtion weight 
        omega is the inter-module connection weight of the neuron
        """
        self.alpha = weights[0]
        self.omega = weights[1]
        
    def get_action(self, dt):
        """
        Return action based off of observation and current CPG params
        
        """
        num_neurons = 2
        def ode_to_solve(state, tau, E, B, alpha, omega, y_other_neuron, y_prev_mod):
            U, V= state
            y = max(0, U)
            dUdt = (E - B * V - alpha * y_other_neuron + omega * y_prev_mod - U)/tau
            dVdt = (y - V)/tau
            return [dUdt, dVdt]
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        action = np.zeros((self.num_mods))
        tau = self.params[0]
        Bs = self.params[1:self.num_mods + 1]
        Es = self.params[self.num_mods + 1:]
        back_leg = [0, 1, 2]
        front_leg = [3, 4, 5]
        for i in range(self.num_mods):
            # print(i)
            E = Es[i]
            B = Bs[i]
            if self.fix_B:
                Bs[0] = self.B1
            for j in range(num_neurons):
                if i != 0: 
                    y_prev_mod = self.y[i-1, j]
                else:
                    y_prev_mod = 0.0
                if i == 3:
                    y_prev_mod = 0
                y_other_neuron = self.y[i, 1-j]
                state = [self.U[i,j], self.V[i,j]]
                t_points = [0,dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_to_solve(state, tau, E, B, self.alpha, self.omega, y_other_neuron, y_prev_mod),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.U[i,j] = solution.y[0, 1]
                    self.V[i,j] = solution.y[1, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"y_other_neuron= {y_other_neuron}\n")
                    print(f"y_prev_mod= {y_prev_mod}")
                    print(f"tau= {tau}\n")
                    print(f"E= {E}\n")
                    print(f"B= {B}\n")
                    print(f"dt= {dt}\n")
                    print(f"alpha={self.alpha}\n")
                    print(f"omega={self.omega}\n")
                    pass
                if self.first_time:
                    if self.U[i, j] > 30 or self.V[i, j] > 30:
                        print("PAST A THRESHOLD\n")
                        print(f"state= {state}\n")
                        print(f"y_other_neuron= {y_other_neuron}\n")
                        print(f"y_prev_mod= {y_prev_mod}")
                        print(f"tau= {tau}\n")
                        print(f"E= {E}\n")
                        print(f"B= {B}\n")
                        print(f"dt= {dt}\n")
                        print(f"alpha={self.alpha}\n")
                        print(f"omega={self.omega}\n")
                        self.first_time=False
                    # self.first_time=False
                self.y[i, j] = max(0, self.U[i,j])
            y_out = self.y[i, 0] - self.y[i, 1]
            action[i] = y_out
        return action

    def run(self, 
            env,
            max_episode_length=60):
        """Run an episode.

        Args:
            env: The env object we need to run a step
            max_episode_length: The maximum time window we will allow for
                interactions within a single episode (i.e 2 seconds, 3 seconds, etc.)
                Default is 2 seconds because a typical turtle gait lasts for about that long.
        Returns:
            A tuple (cumulative_reward, total_episode_time).
        """

        # TODO: look into whether you need to normalize the observation or not
        cumulative_reward = 0.0
        observation, __ = env.reset()
        t = 0
        first_time = True
        total_actions = np.zeros((self.num_mods, 1))
        while True:
            dt = 0.05
            action = self.get_action(dt)
            # print(f"action shape: {action.shape}")
            total_actions = np.append(total_actions, action.reshape((6,1)), axis=1)
            # print(f"params: {self.params}\n")
            # print(f"action: {action}\n")
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            cumulative_reward += reward
            # print(info)
            # t = time.time()
            # print(f"reward and cost: {(info['reward_run'], info['reward_ctrl'])}")
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
        return cumulative_reward, total_actions
    def set_params_and_run(self,
                           env,
                           policy_parameters: ParamVector,
                           max_episode_length=60,
                           ):
        """Set the the parameters of the policy by copying them
        from the given parameter vector, then run an episode.

        Args:
            policy_parameters: The policy parameters to be used.
            decrease_rewards_by: The reward at each timestep will be
                decreased by this given amount.
            max_episode_length: The maximum number of interactions
                allowed in an episode.
        Returns:
            A tuple (cumulative_reward, t (len of episode in seconds)).
        """
        self.set_parameters(policy_parameters)
        
        cumulative_reward, total_actions = self.run(env,
            max_episode_length=max_episode_length
        )
        return cumulative_reward, total_actions

def save_run(folder_name, i):
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(folder_name + f'CPG_output_run{i}.png')

def train(num_params=20, num_mods=10, M=20, K=3):
    """
    Implements EPHE algorithm, where at each episode we sample params from N(v|h) M times.
    We then choose the best K params that are selected from the sorted reward of R(v^m).
    
    """
    trial_folder = 'CPG_exp_43'

    # env = gym.make(ENV_NAME, render_mode='human')
    env = gym.make(ENV_NAME)
    sca = 1
    mu = np.random.rand((num_params)) * 0.1
#     mu = np.array([0.98076661, 0.14970843, 0.01374643, 1.66121233, 0.03077495, 1.70359801,
#  0.12589284, 0.67640131, 1.73801159, 0.86343889 ,0.81584364 ,1.26293396,
#  0.50965711]
# )
    params = np.random.rand((num_params)) * 0.1
    # params = np.array([0.00707745, 
    #                    2.44747663,
    #                    1.31816959, 
    #                    1.32267618, 
    #                    4.63262177, 
    #                    1.52837503,
    #                    1.09880745, 
    #                    1.06433165, 
    #                    1.39569902, 
    #                    1.57176697, 
    #                    0.63726193, 
    #                    1.2837348,
    #                    0.9913531])
    # sigma = np.ones((num_params)) 
#     sigma = np.array([0.09381733, 
#                       0.08027223,
#                         0.01558068,
#                           0.08283432,
#                             0.00635162,
#                               0.09244242,
#                         0.02042298,
#                           0.05694094,
#                             0.0149558,
#                                 0.05940697,
#                                   0.06112813,
#                                     0.05143547,
#  0.0815521 ])
    sigma = np.random.rand((num_params)) + 0.3

    print(f"intial mu: {mu}\n")
    print(f"initial sigma: {sigma}\n")
    print(f"M: {M}\n")
    print(f"K: {K}\n")

    alpha = 0.5
    omega = 0.5
    cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=alpha, omega=omega)
    cpg.set_parameters(params=params)

    print(f"starting params: {cpg.get_params()}")
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
    max_episodes = 20
    max_episode_length = 60     # 60 * 0.05 = ~3 seconds

    config_log = {
        "mu_init": list(mu),
        "sigma_init": list(sigma),
        "params": list(params),
        "M": M,
        "K": K,
        "max_episode_length": max_episode_length,
        "alpha": alpha,
        "omega": omega
    }

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

    # Save the dictionary as a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(config_log, json_file)
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
            timesteps = total_actions.shape[1] - 1
            dt = 0.05
            t = np.arange(0, timesteps*dt, dt)
            if fitness > best_reward:
                print(f"params with best fitness: {solutions[i]} with reward {fitness}\n")
                best_reward = fitness
                best_params = solutions[i]
            #     print(f"fitness: {fitness} at trajectory {i}")
            #     subplot=True
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
    torch.save(best_params, 'best_params_ephe.pth')
    # save data structs to matlab 
    scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data, 'param_data': param_data, 'reward_data': reward_data})

    return best_params
    # return 0

def main(args=None):
    best_params = train(num_params=13, num_mods=6, M=40, K=3)
    
if __name__ == '__main__':
    main()
