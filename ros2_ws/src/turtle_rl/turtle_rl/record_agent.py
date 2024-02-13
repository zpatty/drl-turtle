import gymnasium as gym
import torch
import pickle
import numpy as np
from pgpelib import PGPE
from pgpelib.policies import LinearPolicy, MLPPolicy
from pgpelib.restore import to_torch_module

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

ENV_NAME = 'HalfCheetah-v4'
AGENT = 'cheetah-agent'
PARAM_FILE = 'best_params.pth'
POLICY_FILE = 'policy.pkl'

def load_params(fname):
    best_params = torch.load(fname)
    return best_params

def save_policy(policy):
    with open('policy.pkl', 'wb') as f:
        pickle.dump(policy, f)

def load_policy(fname):
    with open(fname, 'rb') as f:
        policy = pickle.load(f)
    return policy

def test(policy, best_params, num_eval_episodes=2):

    # instantiate gym environment 
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    # set recording thing
    env = RecordVideo(env, video_folder=AGENT, name_prefix="eval",
                  episode_trigger=lambda x: True)

    rewards = []
    # load parameters of final solution into the policy
    policy.set_parameters(best_params)
    # convert policy object to a PyTorch module
    net = to_torch_module(policy)
    # Now we test out final policy
    # Declare the cumulative_reward variable, which will accumulate
    # all the rewards we get from the environment
    for episode in range(num_eval_episodes):
        # Reset the environment, and get the observation of the initial
        # state into a variable.
        observation, __ = env.reset()
        # Visualize the initial state
        env.render()
        cumulative_reward = 0.0

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
            episode_done = truncated or terminated
            env.render()
            cumulative_reward += reward
            print("...\n")
            if episode_done:
                print("DONE")
                rewards.append(cumulative_reward)
                break    
    return rewards

best_params = load_params(PARAM_FILE)
policy = load_policy(POLICY_FILE)
rewards = test(policy=policy, best_params=best_params, num_eval_episodes=5)
print(f"rewards from learned policy: {rewards}")