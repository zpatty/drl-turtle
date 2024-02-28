from pgpelib import PGPE
from pgpelib.policies import LinearPolicy, MLPPolicy
from pgpelib.restore import to_torch_module

from scipy.stats import truncnorm
import numpy as np
import torch

import gym

ENV_NAME = 'CartPole-v1'

class EPHE:
    """
    The EM-based Policy Hyper- Parameter Exploration (EPHE) algorithm.

    Reference Paper: https://link.springer.com/article/10.1007/s10015-015-0260-7
        Jiexin Wang 1 · Eiji Uchibe2 · Kenji Doya
        EM-based policy hyper parameter exploration: application
        to standing and balancing of a two-wheeled smartphone robot
    """
    def __init__(self,
                 solution_length = 10, 
                 popsize=20, 
                 center_init=0.1, 
                 stdev_init=0.1, 
                 dtype="float32",
                 K=3,
                 seed=None):
        """
        Args:
            solution_length: Length of parameters to optimize for
            popsize: Population size
            center_init: Initial mean (center sol)
            stdev_init: Initial standard deviation
            seed: to set the random seed for stochastic step
        
        """
        self._length = solution_length
        self._popsize = popsize  
        self._mu = center_init
        self._sigma = stdev_init
        self.K = K
        if isinstance(dtype, str):
            self._dtype = np.dtype(dtype)
        else:
            self._dtype = dtype
    def _grab_from_distribution(self, truncated=True):
        """
        Grab from truncated normal distribution 
        """
        solutions = []
        
        if truncated:
            lower_bound = 0 
            upper_bound = np.inf
            for i in range(self._popsize):
                mu = self._mu
                sigma = self._sigma
                a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
                normal_dist = truncnorm(a, b, loc=self._mu, scale=self._sigma)
                solution = normal_dist.rvs(size=self._length).astype(self._dtype)
                solutions.append(solution)
        else:
            print("TODO: implement this")
        return solutions
    def ask(self):
        """
        Ask for a batch of solutions to evalute.

        Returns: List of numpy array solutions
        """
        return self._grab_from_distribution()
    def update(self, k_rewards, k_params):
        """
        Update the new mu and sigma from k best rewards and corresponding params
        """
        k_sum = 0
        for k in range(self.K):
            k_sum += k_rewards[k] * k_params[:,k]
        new_mu = k_sum/np.sum(k_rewards)

        sig_sum = 0
        mu = self._mu.copy()
        for k in range(self.K):
            sig_sum += k_rewards[k] * (k_params[:,k] - mu[k])**2
        print(f"sig sum: {sig_sum}\n")
        print(f"reward sum: {np.sum(k_rewards)}\n")
        new_sigma = np.sqrt(sig_sum/np.sum(k_rewards))
        self._mu = new_mu
        self._sigma = new_sigma
    def grab_params(self):
        """
        Returns params from current distribution generated from current mu and sigma 
        """
        a_trunc = 0
        b_trunc = np.inf
        a, b = (a_trunc - self._mu) / self._sigma, (b_trunc - self._mu) / self._sigma
        normal_dist = truncnorm(a, b, loc=self._mu, scale=self._sigma)
        params = normal_dist.rvs(size=self._length).astype(self._dtype)
        return params

    def center(self):
        return self._mu
    def sigma(self):
        return self._sigma
        

def train(num_params=20, num_mods=10, M=20, K=3):

    # our initial solution (initial parameter vector) for EPHE to start exploring from 
    mu = np.random.rand((num_params)) * 10
    sigma = np.ones((num_params)) * 0.3

    print(f"initial solution: {mu}")

    ephe = EPHE(
        
        # We are looking for solutions whose lengths are equal
        # to the number of parameters required by the policy:
        solution_length=mu.shape[0],
        
        # Population size:
        popsize=200,
        
        # Initial mean of the search distribution:
        center_init=mu,
        
        # Initial standard deviation of the search distribution:
        stdev_init=0.08,

        # dtype is expected as float32 when using the policy objects
        dtype='float32'
    )

    # Number of iterations (a.k.a sampled rollouts)
    num_iterations = 15
    # The main loop of the evolutionary computation
    for i in range(1, 1 + num_iterations):

        # Get the solutions from the ephe solver
        solutions = ephe.ask()
        fitnesses = []
        
        for solution in solutions:

            fitness, num_interactions = cpg.set_params_and_run(env, solution)
            fitnesses.append(fitness)
        
        # We inform our pgpe solver of the fitnesses we received,
        # so that the population gets updated accordingly.
        ephe.tell(fitnesses)
        
        print("Iteration:", i, "  median score:", np.median(fitnesses))

    # center point (mean) of the search distribution as final solution
    best_params = ephe.center.copy()
    return best_params


def main(args=None):
    """
    Basic EPHE example using the Cartpole-v1 environment. This is to double check that the general algorithm works. 
    """
    return 0
    
if __name__ == '__main__':
    main()
