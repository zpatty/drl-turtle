import os
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


import numpy as np
import pickle
import torch

import gymnasium as gym
from EPHE import EPHE

ParamVector = Union[List[Real], np.ndarray]
Action = Union[List[Real], np.ndarray, Integral]
# from gym.version import VERSION
# print(f"VERSION: {VERSION}")

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


class CoupledCPG:
    """
    Ijspeert implementation
    paper references: https://link.springer.com/article/10.1007/s10514-007-9071-6
    """
    def __init__(self, 
                 num_params=21,
                 num_mods=10,
                 alpha=0.3,
                 omega=0.3,
                 observation_normalization=True,
                 seed=0.0):
        # self._observation_space, self._action_space = (
        #     get_env_spaces(self._env_name, self._env_config)
        # )

        self._seed = seed                           # seed to replicate 
        self.alpha = alpha                          # mutual inhibition weight
        self.omega = omega                          # inter-module connection weight of the neuron
        self.num_mods = num_mods                    # number of CPG modules
        self.y = np.zeros((num_mods, 2))            # holds the output ith CPG mod (2 neurons per cpg mod)
        self.params = np.random.rand((num_params))  # holds the params of the CPG [tau, B, E] = [freq, phase, amplitude]
        self.U = np.zeros((num_mods, 2))            # the U state of the CPG
        self.V = np.zeros((num_mods, 2))            # the V state of the CPG
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
        self.params = params
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
        Es = self.params[1:self.num_mods + 1]
        Bs = self.params[self.num_mods + 1:]
        for i in range(self.num_mods):
            E = Es[i]
            B = Bs[i]
            for j in range(num_neurons):
                if i != 0: 
                    y_prev_mod = self.y[i-1, j]
                else:
                    y_prev_mod = 0.0
                y_other_neuron = self.y[i, 1-j]
                state = [self.U[i,j], self.V[i,j]]
                # print(f"state: {state}\n")
                t_points = [0,dt]

                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_to_solve(state, tau, E, B, self.alpha, self.omega, y_other_neuron, y_prev_mod),
                    t_span=[0, dt], 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                # print(solution)
                try:
                    self.U[i,j] = solution.y[0, 1]
                    self.V[i,j] = solution.y[1, 1]
                except:
                    print("failed to solve ode")
                    pass
                self.y[i, j] = max(0, self.U[i,j])
            y_out = self.y[i, 0] - self.y[i, 1]
            action[i] = y_out
        # print(f"action: {action}")
        return action
    def run(self, 
            env,
            max_episode_length=2.0):
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
        t_start = time.time()
        # t = time.time()
        t = 0
        first_time = True
        max_episode_length = 200
        while True:
            if first_time:
                # dt = 0.0001
                dt = 0.01       # documentation says dt = 0.05 for half cheetah 
                first_time = False
            else:
                # dt = 0.0001
                dt = 0.01
                # print(dt)
            action = self.get_action(dt)
            print(f"action: {action}\n")
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            time_elapsed = time.time() - t_start
            cumulative_reward += reward
            # t = time.time()
            print(f"reward and cost: {(info['reward_run'], info['reward_ctrl'])}")
            t += 1
            if t > max_episode_length:
                break
            if done:
                break
        return cumulative_reward, t
    def set_params_and_run(self,
                           env,
                           policy_parameters: ParamVector,
                           max_episode_length=2.0,
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
        cumulative_reward, t = self.run(env,
            max_episode_length=max_episode_length
        )
        return cumulative_reward, t





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
        self.y = np.random.rand(num_mods, 2) * 0.1          # holds the output ith CPG mod (2 neurons per cpg mod)
        self.B1 = B1                                        # the first phase is fixed                             
        self.params = np.random.rand((num_params)) * 0.1    # holds the params of the CPG [tau, B, E] = [freq, phase, amplitude]
        self.U = np.random.rand(num_mods, 2) * 0.1          # the U state of the CPG
        self.V = np.random.rand(num_mods, 2) * 0.1          # the V state of the CPG
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
        tau = scaled[0] * 100
        scaled[1:self.num_mods + 1]= (scaled[1:self.num_mods + 1]) % (2*np.pi)
        scaled[0] = tau
        # # scaled[1:self.num_mods + 1] = scaled[1:self.num_mods + 1]/tau
        # scaled[self.num_mods + 1:] = scaled[self.num_mods + 1:]/tau
        if scaled[0] < 0:
            for i in range(10):
                print(f"ITS NEGATIVE\n")
        self.params = scaled
    
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
                    # if i==3 and i-1 not in front_leg:
                    #     y_prev_mod = 0
                else:
                    y_prev_mod = 0.0
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
                # print(solution)
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
        # print(f"action: {action}")
        return action
    def run(self, 
            env,
            max_episode_length=2.0):
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
        t_start = time.time()
        # t = time.time()
        t = 0
        first_time = True
        max_episode_length = 750
        while True:
            # dt = 0.01
            # dt = 0.005    
            dt = 0.05
            action = self.get_action(dt)
            # print(f"params: {self.params}\n")
            # print(f"action: {action}\n")
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            time_elapsed = time.time() - t_start
            cumulative_reward += reward
            # print(info)
            # t = time.time()
            # print(f"reward and cost: {(info['reward_run'], info['reward_ctrl'])}")
            t += 1
            # if t > max_episode_length:
            #     break
            if done:
                break
        return cumulative_reward, t
    def set_params_and_run(self,
                           env,
                           policy_parameters: ParamVector,
                           max_episode_length=2.0,
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
        
        cumulative_reward, t = self.run(env,
            max_episode_length=max_episode_length
        )
        return cumulative_reward, t


def train(num_params=20, num_mods=10, M=20, K=3):
    """
    Implements EPHE algorithm, where at each episode we sample params from N(v|h) M times.
    We then choose the best K params that are selected from the sorted reward of R(v^m).
    
    """
    # env = gym.make(ENV_NAME, render_mode='human')
    env = gym.make(ENV_NAME, ctrl_cost_weight=0.01)

    mu = np.random.rand((num_params)) * 2
    # mu = np.array([7.07744993e-03, 
    #             2.44747663e+00,
    #             1.31816959e+00,
    #             1.32267618e+00,
    #             4.63262177e+00,
    #             1.52837503e+00,
    #             1.09880745e+00,
    #             1.06433165e+00,
    #             1.39569902e+00,
    #             1.57176697e+00,
    #             6.37261927e-01,
    #             1.28373480e+00,
    #             9.91353095e-01])
    # sigma = np.ones((num_params)) * 0.8
    sigma = np.random.rand((num_params)) * 0.4

    print(f"intial mu: {mu}\n")
    print(f"initial sigma: {sigma}\n")
    print(f"M: {M}\n")
    print(f"K: {K}\n")


    cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=0.3, omega=0.5)
    # cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=2.5, omega=1.0)
    # cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=0.3, omega=0.5, B1=15, fix_B=True)
    # Bo Chen paper says robot converged in 20 episodes
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

    max_episodes = 20
    for episode in range(max_episodes):

        # The main loop of the evolutionary computation
        # this is where we run our M trajectories
        lst_params = np.zeros((num_params, M))
        # Get the M solutions from the ephe solver
        solutions = ephe.ask()          # this is population size
        R = np.zeros(M)
        for i in range(M):
            if min(solutions[i]) < 0:
                print(f" sol: {solutions[i]}")
            lst_params[:, i] = solutions[i]
            fitness, num_interactions = cpg.set_params_and_run(env, solutions[i])
            if fitness < 0:
                R[i] = 0
            else:
                R[i] = fitness
        
        print("Episode:", episode, "  median score:", np.median(R))
        print(f"all rewards: {R}\n")
        # get indices of K best rewards
        best_inds = np.argsort(-R)[:K]
        # print(f"best inds: {best_inds}")
        k_params = lst_params[:, best_inds]
        print(f"k params: {k_params}")
        k_rewards = R[best_inds]
        print(f"k rewards: {k_rewards}")
        # We inform our ephe solver of the fitnesses we received,
        # so that the population gets updated accordingly.
        ephe.update(k_rewards=k_rewards, k_params=k_params)
        print(f"new mu: {ephe.center()}\n")
        print(f"new sigma: {ephe.sigma()}\n")
        
    best_mu = ephe.center()
    best_sigma = ephe.sigma()
    print(f"best mu: {best_mu}\n")
    print(f"best sigma: {best_sigma}\n")

    best_params = ephe.grab_params()
    print(f"best params: {best_params}\n")
    # save the best params
    torch.save(best_params, 'best_params_ephe.pth')

    return best_params

def main(args=None):
    best_params = train(num_params=13, num_mods=6, M=20, K=3)
    
if __name__ == '__main__':
    main()
