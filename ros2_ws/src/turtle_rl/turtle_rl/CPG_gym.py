
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CPGGym(gym.Env):
    """
    Sim gym to test CPG output on small networks 
    """

    def __init__(self, robot, cpg, max_episode_length, info):
        super().__init__()
        self.robot = robot
        self.cpg = cpg
        if info["robot"] == "half-cheetah":
            self.r_low = -2
            self.r_high = 2

            self.x_low = -1
            self.x_high = 1

            # self.w_low = 2 * np.pi * 0.4
            # self.w_high = 2 * np.pi * 5


            self.w_low = 0.4
            self.w_high = 5

            self.action_space = spaces.Box(low=np.array([self.w_low, 
                                                        self.r_low, self.r_low, self.r_low, self.r_low, self.r_low, self.r_low,
                                                        self.x_low, self.x_low, self.x_low, self.x_low, self.x_low, self.x_low]),
                                            high=np.array([self.w_high, 
                                                        self.r_high, self.r_high, self.r_high, self.r_high, self.r_high, self.r_high,
                                                        self.x_high, self.x_high, self.x_high, self.x_high, self.x_high, self.x_high]))
            print(f"action space: {self.action_space}")
        else:
            self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        self.max_episode_length = max_episode_length

    def step(self, action):
        """
        Action refers to the new set of CPG parameters
        For this version we're going to have it pass the CPG parameters and we'll run that as the action 
        @action in this case are the CPG params
        @ we do one time step of it moving given the new CPG params
        """
        # need to clip action for some reason
        action = np.clip(action, np.array([self.w_low, 
                                                        self.r_low, self.r_low, self.r_low, self.r_low, self.r_low, self.r_low,
                                                        self.x_low, self.x_low, self.x_low, self.x_low, self.x_low, self.x_low]), np.array([self.w_high, 
                                                        self.r_high, self.r_high, self.r_high, self.r_high, self.r_high, self.r_high,
                                                        self.x_high, self.x_high, self.x_high, self.x_high, self.x_high, self.x_high]) )
        print(f"action {action}")
        return self.cpg.set_params_and_step(env=self.robot, policy_parameters=action, max_episode_length=self.max_episode_length, PD=True)
        # return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        print("reseting...")
        self.cpg.reset()
        observation, info = self.robot.reset()
        return observation, info

    def render(self):
        return self.robot.render()

    def close(self):
        return self.robot.close()
