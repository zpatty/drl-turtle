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
from AukeCPG import AukeCPG

import numpy as np
import pickle
import torch

import gymnasium as gym

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
seed = 0
set_random_seed(seed=seed)

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

def train(num_params=20, num_mods=10, save=False):
    # env = gym.make(ENV_NAME, render_mode="human")
    print("made env")
    env = gym.make(ENV_NAME)
    # our initial solution (initial parameter vector) for PGPE to start exploring from 
    x0 = np.zeros((num_params))
    # mu = np.zeros((num_params))
    # sigma = 0.3
    # mu = np.random.rand((num_params)) 
    # mu[0] = 7.5
    mu = np.array([4.5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    sigma = np.random.rand((num_params)) + 0.3
    # sigma = np.random.random((num_params)) * 0.7
    sigma[0] = 0.1
    print(f"initial solution: {x0}")
    phi=0.0
    w=0.5
    a_r=25
    a_x=25
    dt = 0.05
    # cpg = DualCPG(num_params=num_params, num_mods=num_mods)
    cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)
    
    pgpe = PGPE(

        
        # We are looking for solutions whose lengths are equal
        # to the number of parameters required by the policy:
        solution_length=x0.shape[0],
        
        # Population size:
        popsize=50,
        
        # Initial mean of the search distribution:
        center_init=mu,
        
        # Learning rate for when updating the mean of the search distribution:
        center_learning_rate=0.25,
        
        # Optimizer to be used for when updating the mean of the search
        # distribution, and optimizer-specific configuration:
        # optimizer='adam',
        optimizer='clipup',
        optimizer_config={'max_speed': 0.15},
        
        # Initial standard deviation of the search distribution:
        stdev_init=sigma,
        
        # Learning rate for when updating the standard deviation of the
        # search distribution:
        stdev_learning_rate=0.1,
        
        # Limiting the change on the standard deviation:
        stdev_max_change=0.3,
        
        # Solution ranking (True means 0-centered ranking will be used)
        solution_ranking=True,
        
        # dtype is expected as float32 when using the policy objects
        dtype='float32'
    )

    print(f"PGPE: {pgpe}")

    # Number of iterations
    num_iterations = 20
    # The main loop of the evolutionary computation
    for i in range(num_iterations):
        print(f"-------------iteration: {i}-------------------")
        # Get the solutions from the pgpe solver
        solutions = pgpe.ask()                      # this is population size
        fitnesses = []
        # print(f"num of sols: {len(solutions)}")
        observation, __ = env.reset()
        for solution in solutions:
            # print(f"params: {solution}\n")
            cpg.set_parameters(solution)
            max_episode_length = 20/0.05
            fitness, total_actions = cpg.set_params_and_run(env=env, policy_parameters=solution, max_episode_length=max_episode_length, PD=True)
            print(f"reward: {fitness}")
            fitnesses.append(fitness)
        # We inform our pgpe solver of the fitnesses we received,
        # so that the population gets updated accordingly.
        pgpe.tell(fitnesses)
        
        print("Iteration:", i, "  median score:", np.median(fitnesses))

    # center point (mean) of the search distribution as final solution
    center_solution = pgpe.center.copy()
    best_params = center_solution

    # save the best params
    torch.save(best_params, 'best_params.pth')
    # if save:
    #     save_policy(policy=policy, fname=POLICY_FILE)
    #     return policy, best_params
    # else:
    #     return best_params
    return best_params
# def test(best_params):

#     # instantiate gym environment 
#     env = gym.make(ENV_NAME, render_mode="human")
#     # load parameters of final solution into the policy
#     # Now we test out final policy
#     # Declare the cumulative_reward variable, which will accumulate
#     # all the rewards we get from the environment
#     cumulative_reward = 0.0

#     # Reset the environment, and get the observation of the initial
#     # state into a variable.
#     observation, __ = env.reset()
#     # Visualize the initial state
#     env.render()

#     # Main loop of the trajectory
#     while True:

#         # We pass the observation vector through the PyTorch module
#         # and get an action vector
#         with torch.no_grad():
#             # action = net(
#             #     torch.as_tensor(observation, dtype=torch.float32)
#             # ).numpy()
#             action = 0

#         if isinstance(env.action_space, gym.spaces.Box):
#             # If the action space of the environment is Box
#             # (that is, continuous), then the action vector returned
#             # by the policy is what we will send to the environment.
#             # This is the case for continuous control environments
#             # like 'Humanoid-v2', 'Walker2d-v2', 'HumanoidBulletEnv-v0'.
#             interaction = action
#         elif isinstance(env.action_space, gym.spaces.Discrete):
#             # If the action space of the environment is Discrete,
#             # then the returned vector is in this form:
#             #   [ suggestionForAction0, suggestionForAction1, ... ]
#             # We get the index of the action that has the highest
#             # suggestion value, and that index is what we will
#             # send to the environment.
#             # This is the case for discrete-actioned environments
#             # like 'CartPole-v1'.
#             interaction = int(np.argmax(action))
#         else:
#             assert False, "Unknown action space"

#         observation, reward, terminated, truncated, info = env.step(interaction)
#         done = truncated or terminated
#         env.render()
#         cumulative_reward += reward
#         print("...\n")
#         if done:
#             print("DONE")
#             break
    
#     return cumulative_reward


def main(args=None):
    best_params = train(num_params=13, num_mods=6)
    # best_params = torch.load('best_params.pth')
    # policy = load_policy()
    # reward = test(policy=policy, best_params=best_params)
    # reward = test(best_params)
    # print(f"reward from learned policy: {reward}")
    
if __name__ == '__main__':
    main()
