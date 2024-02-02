from pgpelib import PGPE
from pgpelib.policies import LinearPolicy, MLPPolicy
from pgpelib.restore import to_torch_module

import numpy as np
import torch

import gym

ENV_NAME = 'CartPole-v1'

def train():

    # our policy object (fitness function)
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

    print(f"policy: {policy}")

    # our initial solution (initial parameter vector) for PGPE to start exploring from 
    x0 = np.zeros(policy.get_parameters_count(), dtype='float32')

    print(f"initial solution: {x0}")

    pgpe = PGPE(
        
        # We are looking for solutions whose lengths are equal
        # to the number of parameters required by the policy:
        solution_length=policy.get_parameters_count(),
        
        # Population size:
        popsize=250,
        
        # Initial mean of the search distribution:
        center_init=x0,
        
        # Learning rate for when updating the mean of the search distribution:
        center_learning_rate=0.075,
        
        # Optimizer to be used for when updating the mean of the search
        # distribution, and optimizer-specific configuration:
        optimizer='clipup',
        optimizer_config={'max_speed': 0.15},
        
        # Initial standard deviation of the search distribution:
        stdev_init=0.08,
        
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
    num_iterations = 50

    # The main loop of the evolutionary computation
    for i in range(1, 1 + num_iterations):

        # Get the solutions from the pgpe solver
        solutions = pgpe.ask()

        # The list below will keep the fitnesses
        # (i-th element will store the reward accumulated by the
        # i-th solution)
        fitnesses = []
        
        for solution in solutions:
            # For each solution, we load the parameters into the
            # policy and then run it in the gym environment,
            # by calling the method set_params_and_run(...).
            # In return we get our fitness value (the accumulated
            # reward), and num_interactions (an integer specifying
            # how many interactions with the environment were done
            # using these policy parameters).
            fitness, num_interactions = policy.set_params_and_run(solution)
            
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
    return policy, best_params

def test(policy, best_params):

    # instantiate gym environment 
    env = gym.make(ENV_NAME)

    # load parameters of final solution into the policy
    policy.set_parameters(best_params)
    # convert policy object to a PyTorch module
    net = to_torch_module(policy)



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
            action = net(
                torch.as_tensor(observation, dtype=torch.float32)
            ).numpy()

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

        if done:
            break
    
    return cumulative_reward


def main(args=None):
    policy, best_params = train()
    reward = test(policy=policy, best_params=best_params)
    print(f"reward from learned policy: {reward}")
    
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