import os
import sys
import torch
import time
import scipy
from torch import nn
from copy import copy, deepcopy
import numpy as np
from EPHE import EPHE
from typing import Optional, Union, Iterable, List, Dict, Tuple, Any
from numbers import Real, Integral
# from pgpelib import PGPE
# from pgpelib.policies import LinearPolicy, MLPPolicy
# from pgpelib.restore import to_torch_module
import matplotlib.pyplot as plt

import numpy as np
import pickle
import torch

# import gymnasium as gym

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)
ParamVector = Union[List[Real], np.ndarray]
Action = Union[List[Real], np.ndarray, Integral]

ENV_NAME = 'HalfCheetah-v4'
PARAM_FILE = 'best_params.pth'
POLICY_FILE = 'policy.pkl'

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
                 dt=0.002,
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
        # self.U = np.random.rand(num_mods, 2) * 10          # the U state of the CPG
        # self.V = np.random.rand(num_mods, 2) * 10          # the V state of the CPG
        # self.U = np.random.uniform(low=10, high=50, size=(num_mods,2))
        # self.V = np.random.uniform(low=10, high=50, size=(num_mods,2))

        self.U = np.zeros((num_mods, 2))
        self.V = np.zeros((num_mods, 2))
        self.fix_B = fix_B                                  # whether or not we fix the first B1 like in BoChen
        self.first_time = True
        self.dt = dt
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
        # print(f"scale: {type(scaled)}")
        # scale the phase
        # scaled[1:self.num_mods + 1]= (scaled[1:self.num_mods + 1]) % (2*np.pi)
        self.params = params
        print(f"-----------------current params--------------------: {self.params}")
        print(f"starting state: {[self.U, self.V]}")
    def get_params(self):
        return self.params
    
    def reset(self):
        """
        Reset the initial states of the CPG to enable exploration
        """
        self.U = np.random.uniform(low=1, high=10, size=(self.num_mods,2))
        self.V = np.random.uniform(low=1, high=10, size=(self.num_mods,2))

    def set_weights(self, weights):
        """
        Change the intrinsic weights of the CPG modules. 
        alpha is the mutual inhibtion weight 
        omega is the inter-module connection weight of the neuron
        """
        self.alpha = weights[0]
        self.omega = weights[1]
        
    def get_action(self):
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
                t_points = [0,self.dt]
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
                    print(f"dt= {self.dt}\n")
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
                        print(f"dt= {self.dt}\n")
                        print(f"alpha={self.alpha}\n")
                        print(f"omega={self.omega}\n")
                        self.first_time=False
                    # self.first_time=False
                self.y[i, j] = max(0, self.U[i,j])
            y_out = self.y[i, 0] - self.y[i, 1]
            action[i] = y_out
        return action
    

    def get_rollout(self, episode_length=60):
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
        actions = np.zeros((self.num_mods, episode_length))
        tau = self.params[0]
        Bs = self.params[1:self.num_mods + 1]
        Es = self.params[self.num_mods + 1:]
        for e in range(episode_length):
            action = np.zeros((self.num_mods))
            for i in range(self.num_mods):
                E = Es[i]
                B = Bs[i]
                for j in range(num_neurons):
                    if i != 0: 
                        y_prev_mod = self.y[i-1, j]
                    else:
                        y_prev_mod = 0.0
                    if i == 3:
                        y_prev_mod = 0
                    y_other_neuron = self.y[i, 1-j]
                    state = [self.U[i,j], self.V[i,j]]
                    t_points = [0,self.dt]
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
                        print(f"dt= {self.dt}\n")
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
                            print(f"dt= {self.dt}\n")
                            print(f"alpha={self.alpha}\n")
                            print(f"omega={self.omega}\n")
                            self.first_time=False
                        # self.first_time=False
                    self.y[i, j] = max(0, self.U[i,j])
                y_out = self.y[i, 0] - self.y[i, 1]
                action[i] = y_out
            actions[:, e] = action
        return actions

    def get_coupled_rollout(self, episode_length=60):
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
        actions = np.zeros((self.num_mods, episode_length))
        tau = self.params[0]
        Bs = self.params[1:self.num_mods + 1]
        Es = self.params[self.num_mods + 1:]
        for e in range(episode_length):
            action = np.zeros((self.num_mods))
            for i in range(self.num_mods):
                E = Es[i]
                B = Bs[i]
                for j in range(num_neurons):
                    if i != 0: 
                        y_prev_mod = self.y[i-1, j]
                    else:
                        y_prev_mod = 0.0
                    if i == 6 or i==8 or i==3:
                        y_prev_mod = 0
                    y_other_neuron = self.y[i, 1-j]
                    state = [self.U[i,j], self.V[i,j]]
                    t_points = [0,self.dt]
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
                        print(f"dt= {self.dt}\n")
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
                            print(f"dt= {self.dt}\n")
                            print(f"alpha={self.alpha}\n")
                            print(f"omega={self.omega}\n")
                            self.first_time=False
                        # self.first_time=False
                    self.y[i, j] = max(0, self.U[i,j])
                y_out = self.y[i, 0] - self.y[i, 1]
                action[i] = y_out
            actions[:, e] = action
        return actions

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
            dt = 0.002
            action = self.get_action(dt)
            # print(f"action shape: {action.shape}")
            total_actions = np.append(total_actions, action.reshape((10,1)), axis=1)
            # print(f"params: {self.params}\n")
            # print(f"action: {action}\n")
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            cumulative_reward += reward
            t += 1
            if t > max_episode_length:
                print(f"we're donesies with {t}")
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
    

def main(args=None):
    num_params = 21
    num_mods = 10

    cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=0.3, omega=0.3, dt=0.01)

    # mu = np.random.rand((num_params)) * 5000
    mu = np.random.uniform(low=0, high=2*np.pi, size=num_params)
    mu[1 + num_mods:] = np.random.uniform(low=10, high=20)
    # mu[0] = np.random.random(1) * 0.001
    mu[0] = 1.5
    # mu[num_mods + 1:] = mu[num_mods + 1] + 1000
    print(f"init mu: {mu}")
    sigma = np.random.rand((num_params)) + 0.3

    ephe = EPHE(
                
                # We are looking for solutions whose lengths are equal
                # to the number of parameters required by the policy:
                solution_length=mu.shape[0],
                
                # Population size: the number of trajectories we run with given mu and sigma 
                popsize=10,
                
                # Initial mean of the search distribution:
                center_init=mu,
                
                # Initial standard deviation of the search distribution:
                stdev_init=sigma,

                # dtype is expected as float32 when using the policy objects
                dtype='float32', 

                K=2
            )
    solutions = ephe.ask()     
    # for solution in solutions:
    #     eps_len = 1000
    #     cpg.set_parameters(params=solution)
    #     # cpg.reset()
    #     # cpg.plot(timesteps=60)
    #     total_actions = cpg.get_rollout(episode_length=eps_len)
    #     # print(f"action: {total_actions[:, 0:15]}")
    #     # fitness, total_actions = cpg.set_params_and_run(epolicy_parameters=solutions[i], max_episode_length=max_episode_length)
    #     t = np.arange(0, eps_len*cpg.dt, cpg.dt)

    #     fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
    #     for j, ax in enumerate(axs):
    #         ax.plot(t, total_actions[j, :])
    #         ax.set_title(f"CPG {j+1}")
    #         ax.set_xlabel("Time")
    #         ax.set_ylabel("Data")
    #         ax.grid(True)
    #     plt.tight_layout()

    # ps = [6.4333670e-02, 
    #       4.0481272e+00, 4.1443191e+00, 2.6797774e+00, 
    #       1.9581868e-01, 3.4114313e+00, 4.5388846e+00, 
    #       4.4444680e+00, 3.7272689e-01,
    #     4.8481183e+00, 2.9969893e+00, 
    #     9.0043145e+03, 9.0045352e+03, 9.0035449e+03,
    #     9.0041064e+03, 9.0047783e+03, 9.0030430e+03, 
    #     9.0046689e+03, 9.0047539e+03,
    #     9.0055791e+03, 9.0029482e+03]
    eps_len = 700
    for solution in solutions:
        print(f"sol: {100 * solution}")
        cpg.set_parameters(params=solution)
        # cpg.reset()
        # cpg.plot(timesteps=60)
        total_actions = cpg.get_coupled_rollout(episode_length=eps_len)
        # print(f"action: {total_actions[:, 0:15]}")
        # fitness, total_actions = cpg.set_params_and_run(epolicy_parameters=solutions[i], max_episode_length=max_episode_length)
        t = np.arange(0, eps_len*cpg.dt, cpg.dt)

        fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
        for j, ax in enumerate(axs):
            ax.plot(t, total_actions[j, :]*10e4)
            ax.set_title(f"CPG {j+1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Data")
            ax.grid(True)
        plt.tight_layout()

    plt.show()
    # return 0

if __name__ == "__main__":
    main()