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
                 alpha=0.3,
                 omega=0.3,
                 observation_normalization=True,
                 seed=0.0):
        # self._observation_space, self._action_space = (
        #     get_env_spaces(self._env_name, self._env_config)
        # )

        self._seed = seed       # seed to replicate 
        self.alpha = alpha      # mutual inhibition weight
        self.omega = omega      # inter-module connection weight of the neuron
        self.num_mods = num_mods                # number of CPG modules
        self.y = np.zeros((num_mods, 2))        # holds the output ith CPG mod (2 neurons per cpg mod)
        self.params = np.zeros((num_params,1))  # holds the params of the CPG [tau, B, E] = [freq, phase, amplitude]
        self.U = np.zeros((num_mods, 2))        # the U state of the CPG
        self.V = np.zeros((num_mods, 2))        # the V state of the CPG
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
                    y_prev_mod = 0
                y_other_neuron = self.y[i, 1-j]
                state = [self.U[i,j], self.V[i,j]]

                t_0 = 0.0
                t = 0.1
                t_points = [0,dt]

                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_to_solve(state, tau, E, B, self.alpha, self.omega, y_other_neuron, y_prev_mod),
                    t_span=[t_0, t], 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )

                U = solution.y[0, 1]
                V = solution.y[1, 1]
                self.y[i, j] = max(0, U)
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
        t = time.time()
        first_time = True
        while True:
            if first_time:
                dt = 0.0001
                first_time = False
            else:
                dt = 0.0001
                # print(dt)
            action = self.get_action(dt)
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            time_elapsed = time.time() - t_start
            cumulative_reward += reward
            t = time.time()
            if max_episode_length is not None and time_elapsed > max_episode_length:
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

def train(num_params=20, num_mods=10, save=False):
    env = gym.make(ENV_NAME)
    policy = MLPPolicy(
        
        # The name of the environment in which the policy will be tested:
        env_name=ENV_NAME,
        
        # Number of hidden layers:
        num_hidden=1,
        
        # Size of a hidden layer:
        hidden_size=8,
        
        # Activation function to be used in the hidden layers:
        hidden_activation='tanh',
        
        # Whether or not to do online normalization on the observations
        # received from the environments.
        # The default is True, and using observation normalization
        # can be very helpful.
        # In this tutorial, we set it to False just to keep things simple.
        observation_normalization=False
    )

    # our initial solution (initial parameter vector) for PGPE to start exploring from 
    x0 = np.zeros((num_params))
    mu = np.zeros((num_params))
    sigma = 0.08

    print(f"initial solution: {x0}")

    cpg = DualCPG(num_params=num_params, num_mods=num_mods)

    pgpe = PGPE(

        
        # We are looking for solutions whose lengths are equal
        # to the number of parameters required by the policy:
        solution_length=x0.shape[0],
        
        # Population size:
        popsize=10,
        
        # Initial mean of the search distribution:
        center_init=mu,
        
        # Learning rate for when updating the mean of the search distribution:
        center_learning_rate=0.075,
        
        # Optimizer to be used for when updating the mean of the search
        # distribution, and optimizer-specific configuration:
        optimizer='clipup',
        optimizer_config={'max_speed': 0.15},
        
        # Initial standard deviation of the search distribution:
        stdev_init=sigma,
        
        # Learning rate for when updating the standard deviation of the
        # search distribution:
        stdev_learning_rate=0.1,
        
        # Limiting the change on the standard deviation:
        stdev_max_change=0.2,
        
        # Solution ranking (True means 0-centered ranking will be used)
        solution_ranking=True,
        
        # dtype is expected as float32 when using the policy objects
        dtype='float32'
    )

    print(f"PGPE: {pgpe}")

    # Number of iterations
    num_iterations = 20
    # num_iterations = 10

    # The main loop of the evolutionary computation
    for i in range(1, 1 + num_iterations):

        # Get the solutions from the pgpe solver
        solutions = pgpe.ask()          # this is population size

        # The list below will keep the fitnesses
        # (i-th element will store the reward accumulated by the
        # i-th solution)
        fitnesses = []
        # print(f"num of sols: {len(solutions)}")
        for solution in solutions:
            # For each solution, we load the parameters into the
            # policy and then run it in the gym environment,
            # by calling the method set_params_and_run(...).
            # In return we get our fitness value (the accumulated
            # reward), and num_interactions (an integer specifying
            # how many interactions with the environment were done
            # using these policy parameters).
            # print(f"proposed sol: {solution}")
            fitness, num_interactions = cpg.set_params_and_run(env, solution)
            
            # In the case of this example, we are only interested
            # in our fitness values, so we add it to our fitnesses list.
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
    if save:
        save_policy(policy=policy, fname=POLICY_FILE)
        return policy, best_params
    else:
        return best_params

def test(best_params):

    # instantiate gym environment 
    env = gym.make(ENV_NAME, render_mode="human")
    # load parameters of final solution into the policy
    # Now we test out final policy
    # Declare the cumulative_reward variable, which will accumulate
    # all the rewards we get from the environment
    cumulative_reward = 0.0

    # Reset the environment, and get the observation of the initial
    # state into a variable.
    observation, __ = env.reset()
    # Visualize the initial state
    env.render()

    # Main loop of the trajectory
    while True:

        # We pass the observation vector through the PyTorch module
        # and get an action vector
        with torch.no_grad():
            # action = net(
            #     torch.as_tensor(observation, dtype=torch.float32)
            # ).numpy()
            action = 0

        if isinstance(env.action_space, gym.spaces.Box):
            # If the action space of the environment is Box
            # (that is, continuous), then the action vector returned
            # by the policy is what we will send to the environment.
            # This is the case for continuous control environments
            # like 'Humanoid-v2', 'Walker2d-v2', 'HumanoidBulletEnv-v0'.
            interaction = action
        elif isinstance(env.action_space, gym.spaces.Discrete):
            # If the action space of the environment is Discrete,
            # then the returned vector is in this form:
            #   [ suggestionForAction0, suggestionForAction1, ... ]
            # We get the index of the action that has the highest
            # suggestion value, and that index is what we will
            # send to the environment.
            # This is the case for discrete-actioned environments
            # like 'CartPole-v1'.
            interaction = int(np.argmax(action))
        else:
            assert False, "Unknown action space"

        observation, reward, terminated, truncated, info = env.step(interaction)
        done = truncated or terminated
        env.render()
        cumulative_reward += reward
        print("...\n")
        if done:
            print("DONE")
            break
    
    return cumulative_reward


def main(args=None):
    best_params = train(num_params=13, num_mods=6)
    # best_params = torch.load('best_params.pth')
    # policy = load_policy()
    # reward = test(policy=policy, best_params=best_params)
    # reward = test(best_params)
    # print(f"reward from learned policy: {reward}")
    
if __name__ == '__main__':
    main()

# # Let us run the evolutionary computation for 1000 generations
# for generation in range(1000):

#     # Ask for solutions, which are to be given as a list of numpy arrays.
#     # In the case of this example, solutions is a list which contains
#     # 20 numpy arrays, the length of each numpy array being 5.
#     solutions = pgpe.ask()

#     # This is the phase where we evaluate the solutions
#     # and prepare a list of fitnesses.
#     # Make sure that fitnesses[i] stores the fitness of solutions[i].
#     fitnesses = [...]  # compute the fitnesses here

#     # Now we tell the result of our evaluations, fitnesses,
#     # to our solver, so that it updates the center solution
#     # and the spread of the search distribution.
#     pgpe.tell(fitnesses)

# # After 1000 generations, we print the center solution.
# print(pgpe.center)