from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import os
import sys
import json
from typing import Any
import serial

import numpy as np
import torch 
import gymnasium as gym
from gymnasium import spaces

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos

from pgpelib import PGPE

T = TypeVar("T", int, np.ndarray)

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_rl/turtle_rl"
sys.path.append(submodule)

class CPG:
    """
    EPFL(Ijspeert)'s implementation
    paper references: 
        : https://link.springer.com/article/10.1007/s10514-007-9071-6
        : https://arxiv.org/pdf/2211.00458.pdf 
    """
    def __init__(self, 
                 alpha=1.0,
                 timestep=0.03):
        self.test = 0
        self.traj = []
        self.dist_to_goal = []
        self.dt = timestep
    def step(self, q, action):
        """
        Take a step given current coordinates and particular action taken 
        """

    def run_batch_of_trajectories(self, qs):
        pass
    def reward(self, q, goal):
        pass

class DualCPG:
    """
    Bo Chen implementation (the dual-neuron model)
    paper references:
        : https://arxiv.org/pdf/2307.08178.pdf
        : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9636100
    """
    def __init__(self, 
                 alpha=1.0,
                 timestep=0.03):
        self.test = 0
        self.traj = []
        self.dist_to_goal = []
        self.dt = timestep
    def step(self, q, action):
        """
        Take a step given current coordinates and particular action taken 
        """
    def run_batch_of_trajectories(self, qs):
        pass
    def reward(self, q, goal):
        pass

class TurtleCPG(gym.Env):
    """
    This node is responsible for grabbing sensor data for online training of RL algorithm
    For now we're implementing an adapted version of Bo Chen paper: https://arxiv.org/pdf/2307.08178.pdf
    TODO: look into RL algs besides EM-based Policy Hyper-Parameter Exploration (EPHE)
    """

    def __init__(self, dim= None, space= None, ep_length= 100, num_iterations=10, num_params=3, num_actuators=10):
        """
        Turtle Environment that also functions as a ROS2 node

        :param dim: the size of the action and observation dimension you want
            to learn.
        :param space: the action and observation space. Provide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in timesteps
        """

        self.action_space = spaces.Box(low=0.01, high=0.8, shape=(num_params,num_actuators), dtype=np.float32)
        self.spacing = 30
        self.ep_length = ep_length
        self.current_step = 0
        self.num_iterations = num_iterations
        self.num_resets = -1            # Becomes 0 after __init__ exits.
        # subscribe to collect turtle sensor data (i.e accelerometer for reward function)
        self.sensor_sub = self.create_subscription(
            TurtleSensors,
            'turtle_sensors',
            self.sensors_callback,
            10
        )
        # subscribe to collect motor position date (not sure if needed)
        self.motors_sub = self.create_subscription(
            TurtleMotorPos,
            'turtle_motor_pos',
            self.motors_callback,
            10
        )
        # publish motor commands to turtle (specifically q)
        self.motors_pub = self.create_publisher(
            TurtleTraj,
            'turtle_traj',
            10
        )
        self.acc = np.zeros((3,1))
        self.reset()        # TODO: move this somewhere else

    def sensors_callback(self, msg):
        """
        consistently reads from sensors node and updates turtle state accordingly
        """
        acc_x = msg.imu.linear_acceleration.x
        acc_y = msg.imu.linear_acceleration.y
        acc_z = msg.imu.linear_acceleration.z
        self.acc = np.array([acc_x, acc_y, acc_z]).reshape((3,1))

    def reset(self, seed = None, options = None ):
        """
        Need to reset turtle robot to a neutral state hardware-wise?
        """
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state, {}

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length
        return self.state, reward, terminated, truncated, {}
    
    def send_motor_pos(self, q):
        """
        Sends motor positions to the turtle via the turtle_traj topic
        """
        traj = TurtleTraj()
        traj.qd = q
    
    def step_rollout(self):
        pass
    
    def render(self):
        pass

    def is_done(self):
        pass

    def __render_frame(self):
        pass

    def _choose_next_state(self):
        self.state = self.action_space.sample()

    def _get_reward(self, action):
        """
        Reward function is currently grabbing the acceleration 
        """
        return 1.0 if np.all(self.state == action) else 0.0
    
    def _get_obs(self):
        pass

    def _get_info(self):
        pass


class TurtleCPG(Node):
    """
    ROS2 node that looks at actually training or deploying depending on wha thappend
    """
    def __init__(self):
        self.mode_cmd_sub = self.create_subscription(
            String,
            'turtle_mode_cmd',
            self.turtle_rl_callback,
            10
        )
        self.motor_cmd_pub = self.create_publisher(
            String,
            'turtle_state',
            10
        )
        self.test = 0

    def train(self):
        """
        Gets a command from the keyboard to train the turtle, which we do by then creating a 
        CPG Turtle object that is dependent on the kind of reward function or CPG type received in the msg
        """
        msg = String()
        msg.data='rest'
        self.motor_cmd_pub.publish(msg)
        pass

    def test(self):
        """
        Gets a command from the keyboard to deploy trained model onto turtle to evaluate it
        """
        pass
    

   
def main(args=None):
    rclpy.init(args=args)

    cpg_node = TurtleCPG('turtle_sensors')

    rclpy.spin(cpg_node)

    cpg_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()