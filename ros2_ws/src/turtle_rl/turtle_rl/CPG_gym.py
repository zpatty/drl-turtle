
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CPGGym(gym.Env):
    """
    Sim gym to test CPG output on small networks 
    """

    def __init__(self, robot, cpg, max_episode_length):
        super().__init__()
        self.robot = robot
        self.cpg = cpg
        # self.action_space = self.robot.action_space
        self.action_space = spaces.Box(low=0, high=2 * np.pi,
                                            shape=(self.cpg.num_params,), dtype=np.float32)
        self.observation_space = self.robot.observation_space
        self.max_episode_length = max_episode_length

    def step(self, action):
        """
        Action refers to the new set of CPG parameters
        """
        self.cpg.set_parameters(action)
        reward, total_actions = self.cpg.set_params_and_run(env=self.robot, policy_parameters=action, max_episode_length=self.max_episode_length, PD=True)
        observation = "hai"
        terminated = True
        truncated = True
        info = "hi"
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.robot.reset()
        return observation, info

    def render(self):
        return self.robot.render()

    def close(self):
        return self.robot.close()
