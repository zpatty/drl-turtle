import os
import sys
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

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_rl/turtle_rl"
sys.path.append(submodule)
ParamVector = Union[List[Real], np.ndarray]
Action = Union[List[Real], np.ndarray, Integral]

ENV_NAME = 'HalfCheetah-v4'
PARAM_FILE = 'best_params.pth'
POLICY_FILE = 'policy.pkl'

class AukeCPG:
    """
    Auke Ijspeert implementation (the coupled CPG model)
    paper references:
        : https://link.springer.com/article/10.1007/s10514-007-9071-6

    """
    def __init__(self, 
                 num_params=21,
                 num_mods=10,
                 phi=0.0,
                 w=np.random.rand() * np.pi * 2,
                 a_r=20,
                 a_x=20,
                 dt=0.05):

        self.w = w                          # coupled weight bias
        self.phi = phi                      # coupled phase bias- this seems to always be 0 according to auke
        self.num_mods = num_mods            # number of CPG modules
        self.theta = np.zeros((num_mods))   # output of oscillators (radians)

        self.params = np.random.rand((num_params)) * 0.1        # holds the params of the CPG [omega, R, X] = [freq, amplitude, offset]
        self.PHI = np.zeros((num_mods))                         # phase state variables (radians)
        self.r = np.zeros((num_mods))                           # amplitude state variables (radians)
        self.x = np.zeros((num_mods))                           # offset state variables (radians)                           
        self.v = np.zeros((num_mods))                           # to handle second order eq of amplitude
        self.m = np.zeros((num_mods))                           # to handle second order eq of offset
        self.dt = dt
        self.a_r = a_r
        self.a_x = a_x

        print(f"starting phi: {self.phi}\n")
        print(f"starting w: {self.w}\n")
        print(f"number of CPG modules: {self.num_mods}\n")
    def set_parameters(self, params):
        """
        Updates parameters of the CPG oscillators.
        We currently have the structure to be a 21x1 vector like so:
        = omega: frequency for all oscillators
        = R1 : amplitude for CPG mod 2
        = ...       
        = Rn: amplitude for CPG mod n
        = X1 : offset for CPG mod 1
        = ...
        = Xn: offset for CPG mod n
        """
        self.params = params
        # print(f"current params: {self.params}")
    
    def get_params(self):
        return self.params
    
    def reset(self):
        """
        Reset your CPGs?
        """
        self.PHI = np.random.uniform(low=0.5, high=20, size=self.num_mods)
        self.r = np.random.uniform(low=0.5, high=20, size=self.num_mods)
        self.x = np.random.uniform(low=0.5, high=20, size=self.num_mods)
        self.m = np.random.uniform(low=0.5, high=20, size=self.num_mods)
        self.v = np.random.uniform(low=0.5, high=20, size=self.num_mods)

        
    def get_action(self, dt):
        """
        Return action based off of observation and current CPG params
        
        """
        def ode_fin(state, omega, R, X, rs, phis):
            PHI, r, x, v, m= state
            dPhidt = omega
            for i in range(len(len(rs))):
                r_other = rs[i]
                phi_other = phis[i]
                dPhidt += self.w*r_other*np.sin(phi_other - PHI - self.phi)
            dRdt = v
            dVdt = self.a_r * ((self.a_r/4) * (R-r) - dRdt)
            dXdt = m
            dMdt = self.a_x * ((self.a_x/4) * (X-x) - dXdt)
            return [dPhidt, dRdt, dXdt, dVdt, dMdt]
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        action = np.zeros((self.num_mods))
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]
        front_fins = [[0, 1, 2], [3, 4, 5]]
        back_fins = [[6, 7], [8, 9]]
        # for every front fin
        for fin in front_fins:
            num_coupled = 3
            # for each oscillator in front fin
            for f in range(len(fin)):
                # grab the index of the oscillator
                idx = fin[f]
                R = Rs[idx]
                X = Xs[idx]
                # find indices of the other two oscillators coupled to current oscillator
                ind1 = fin[(f + 1)%num_coupled]
                ind2 = fin[(f + 2)%num_coupled]
                rs = [self.r[ind1] , self.r[ind2]]
                phis = [self.PHI[ind1] , self.PHI[ind2]]
                # state of current oscillator
                state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                t_points = [0,self.dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.PHI[idx] = solution.y[0, 1]
                    self.r[idx] = solution.y[1, 1]
                    self.x[idx] = solution.y[2, 1]
                    self.v[idx] = solution.y[3, 1]
                    self.m[idx] = solution.y[4, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"dt= {dt}\n")
                    pass
                # grab output of oscillator i
                self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                action[idx] = self.theta[idx]

        # for every back fin
        for fin in back_fins:
            num_coupled = 2
            for f in range(len(fin)):
                idx = fin[f]                            # grab index of current oscillator
                ind = fin[(f + 1) % num_coupled]        # grab index of other oscillator
                rs = [self.r[ind]]
                phis = [self.PHI[ind]]
                # state of current oscillator
                state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                t_points = [0,self.dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.PHI[idx] = solution.y[0, 1]
                    self.r[idx] = solution.y[1, 1]
                    self.x[idx] = solution.y[2, 1]
                    self.v[idx] = solution.y[3, 1]
                    self.m[idx] = solution.y[4, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"dt= {dt}\n")
                    pass
                # grab output of oscillator i
                self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                action[idx] = self.theta[idx]
        return action

    def get_rollout(self, episode_length=60):
        """
        Calculate the entire rollout
        """
        def ode_fin(state, omega, R, X, rs, phis):
            PHI, r, x, v, m= state
            dPhidt = omega
            for i in range(len(rs)):
                r_other = rs[i]
                phi_other = phis[i]
                dPhidt += self.w*r_other*np.sin(phi_other - PHI - self.phi)
            dRdt = v
            dVdt = self.a_r * ((self.a_r/4) * (R-r) - dRdt)
            dXdt = m
            dMdt = self.a_x * ((self.a_x/4) * (X-x) - dXdt)
            return [dPhidt, dRdt, dXdt, dVdt, dMdt]
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        total_actions = np.zeros((self.num_mods, episode_length))
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]
        front_fins = [[0, 1, 2], [3, 4, 5]]
        back_fins = [[6, 7], [8, 9]]
        for i in range(episode_length):
            action = np.zeros((self.num_mods))

            # for every front fin
            for fin in front_fins:
                num_coupled = 3
                # for each oscillator in front fin
                for f in range(len(fin)):
                    # grab the index of the oscillator
                    idx = fin[f]
                    R = Rs[idx]
                    X = Xs[idx]
                    # find indices of the other two oscillators coupled to current oscillator
                    ind1 = fin[(f + 1)%num_coupled]
                    ind2 = fin[(f + 2)%num_coupled]
                    rs = [self.r[ind1] , self.r[ind2]]
                    phis = [self.PHI[ind1] , self.PHI[ind2]]
                    # state of current oscillator
                    state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                    t_points = [0,self.dt]
                    solution = scipy.integrate.solve_ivp(
                        fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                        t_span=t_points, 
                        y0=state,
                        method='RK45',
                        t_eval = t_points
                    )
                    try:
                        self.PHI[idx] = solution.y[0, 1]
                        self.r[idx] = solution.y[1, 1]
                        self.x[idx] = solution.y[2, 1]
                        self.v[idx] = solution.y[3, 1]
                        self.m[idx] = solution.y[4, 1]
                    except:
                        print("failed to solve ode with the following: \n")
                        print(f"state= {state}\n")
                        print(f"dt= {dt}\n")
                        pass
                    # grab output of oscillator i
                    self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                    action[idx] = self.theta[idx]

            # for every back fin
            for fin in back_fins:
                num_coupled = 2
                for f in range(len(fin)):
                    idx = fin[f]                            # grab index of current oscillator
                    ind = fin[(f + 1) % num_coupled]        # grab index of other oscillator
                    rs = [self.r[ind]]
                    phis = [self.PHI[ind]]
                    # state of current oscillator
                    state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                    t_points = [0,self.dt]
                    solution = scipy.integrate.solve_ivp(
                        fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                        t_span=t_points, 
                        y0=state,
                        method='RK45',
                        t_eval = t_points
                    )
                    try:
                        self.PHI[idx] = solution.y[0, 1]
                        self.r[idx] = solution.y[1, 1]
                        self.x[idx] = solution.y[2, 1]
                        self.v[idx] = solution.y[3, 1]
                        self.m[idx] = solution.y[4, 1]
                    except:
                        print("failed to solve ode with the following: \n")
                        print(f"state= {state}\n")
                        print(f"dt= {dt}\n")
                        pass
                    # grab output of oscillator i
                    self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                    action[idx] = self.theta[idx]
            
            # record action for time step into total_actions struct
            total_actions[:, i] = action

        return total_actions

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

def main(args=None):
    num_params = 21
    num_mods = 10
    cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=0.0, w=0.5, a_r=20, a_x=20, dt=0.001)
    params = np.random.uniform(low=0, high=10, size=num_params)
    # params[0] = tau_init

    eps_len = 3000
    cpg.set_parameters(params=params)
    cpg.reset()
    # cpg.plot(timesteps=60)
    total_actions = cpg.get_rollout(episode_length=eps_len)
    print(f"action: {total_actions[:, 0:15]}")
    # fitness, total_actions = cpg.set_params_and_run(epolicy_parameters=solutions[i], max_episode_length=max_episode_length)
    t = np.arange(0, eps_len*cpg.dt, cpg.dt)

    fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
    for j, ax in enumerate(axs):
        ax.plot(t, total_actions[j, :])
        ax.set_title(f"CPG {j+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Data")
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main()