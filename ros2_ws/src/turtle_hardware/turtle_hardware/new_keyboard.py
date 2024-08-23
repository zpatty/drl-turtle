from __future__ import print_function
import os
import sys
import json
import traceback

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
import numpy as np
from matplotlib import pyplot as plt
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_interfaces.msg import TurtleTraj, TurtleSensors
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from turtle_interfaces.msg import TurtleCtrl, TurtleMode

import turtle_trajectory

class TurtleRemote(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, params=None):
        super().__init__('turtle_remote_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )


        timer_cb_group = None
        self.call_timer = self.create_timer(0.05, self._config_cb, callback_group=timer_cb_group)

        # self.handler = ParameterEventHandler(self)


        # continously publishes the current motor position      
        self.config_pub = self.create_publisher(
            TurtleCtrl,
            'turtle_ctrl_params',
            qos_profile
        )

        # continously publishes the current motor position      
        self.mode_pub = self.create_publisher(
            TurtleMode,
            'turtle_mode',
            qos_profile
        )
        # self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(100)
        params, __ = self.parse_ctrl_params()
        print(params)

    def parse_ctrl_params(self):
        with open('ctrl_config.json') as config:
            param = json.load(config)
            self._last_update = os.fstat(config.fileno()).st_mtime
        # print(f"[MESSAGE] Config: {param}\n")    
        # Serializing json
        config_params = json.dumps(param, indent=14)
        return param, config_params
    
    def _config_cb(self):
        if self._last_update != os.stat('ctrl_config.json').st_mtime:
            params, config_params = self.parse_ctrl_params()

            cfg_msg = TurtleCtrl()
            mode_msg = TurtleMode()
            print(params)
            if params["mode"] == " " or params["mode"] == "":
                mode_msg.mode = "rest"
            else:
                mode_msg.mode = params["mode"]

            mode_msg.traj = params["traj"]

            cfg_msg.kp = params["kp"]
            # print("HJere")
            cfg_msg.kd = params["kd"]

            # these should be messages instead of parameters
            cfg_msg.amplitude = params["amplitude"]
            cfg_msg.center = params["center"]
            cfg_msg.yaw = params["yaw"]
            cfg_msg.pitch = params["pitch"]
            cfg_msg.frequency_offset = params["frequency_offset"]
            cfg_msg.period = params["period"]



            cfg_msg.kpv = params["kpv"]
            cfg_msg.d_pinv = params["d_pinv"]
            cfg_msg.learned = bool(params["learned"])
            cfg_msg.kp_s = params["kp_s"]
            cfg_msg.w_th = params["w_th"]
            cfg_msg.kp_th = params["kp_th"]
            cfg_msg.offset = params["offset"]
            cfg_msg.sw = params["sw"]
            print(cfg_msg)
            self.config_pub.publish(cfg_msg)
            self.mode_pub.publish(mode_msg)
            self.get_logger().info('Updated Config')
            t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            with open("data/" + t + "_config.json", 'w') as config:
                json.dump(params, config, ensure_ascii=False, indent=4)

            

if __name__ == '__main__':
    rclpy.init()
    remote_node = TurtleRemote()
    try:
        rclpy.spin(remote_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # rclpy.shutdown()
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
