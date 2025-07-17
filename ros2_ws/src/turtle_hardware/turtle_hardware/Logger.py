import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import sys
import os
from rclpy.parameter_event_handler import ParameterEventHandler

from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode
from std_msgs.msg import String, Float32MultiArray

import numpy as np

from statistics import mode as md

import time

from datetime import datetime
# from turtle_dynamixel.torque_control import torque_control  


class Logger(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, params=None):
        super().__init__('log_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )
        buff_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

                # subscribes to keyboard setting different turtle modes 
        self.ctrl_cmd_sub = self.create_subscription(
            TurtleCtrl,
            'turtle_ctrl_params',
            self.turtle_ctrl_callback,
            qos_profile)
        
        self.mode_cmd_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        # for case when trajectory mode, receives trajectory msg
        self.sensors_sub = self.create_subscription(
            TurtleSensors,
            'turtle_sensors',
            self.sensors_callback,
            qos_profile
        )

        # self.u_sub = self.create_subscription(
        #     TurtleTraj,
        #     'turtle_u',
        #     self.u_callback,
        #     qos_profile
        # )

        self.create_rate(100)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.traj_str = 'sin'


        self.q_data = []
        self.dq_data = []
        self.u_data = []
        self.input_data = []
        self.timestamps = []
        self.qd_data = []
        self.dqd_data = []

        self.amplitude_data = []
        self.center_data = []
        self.yaw_data = []
        self.pitch_data = []
        self.fwd_data = []
        self.roll_data = []
        self.freq_offset_data = []
        self.period_data = []
        self.q_off_data = []
        self.sw_data = []
        self.kpv_data = []
        self.learn_data = []
        self.d_pinv_data = []
        self.kp_th_data = []
        self.w_th_data = []
        self.kp_s_data = []

        self.amplitude = []
        self.center = []
        self.yaw = []
        self.pitch = []
        self.fwd = []
        self.roll = []
        self.freq_offset = []
        self.period = []
        self.q_off = []
        self.sw = []
        self.kpv = []
        self.learn = []
        self.d_pinv = []
        self.kp_th = []
        self.w_th = []
        self.kp_s = []

        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.folder_name =  "data/" + t
        # os.makedirs(self.folder_name)

    def sensors_callback(self, msg):
        """
        Callback function that takes in list of squeezed arrays
        msg: [quat acc gyr voltage t_0 q dq ddq u qd t]
        """    
        # self.get_logger().info(f"Received update to q")
        self.q_data.append(msg.q)
        self.dq_data.append(msg.dq)
        self.qd_data.append(msg.qd)
        self.dqd_data.append(msg.dqd)
        self.u_data.append(msg.u)
        self.input_data.append(msg.input)
        self.timestamps.append(msg.t)

        # control variables
        self.amplitude_data.append(self.amplitude)
        self.center_data.append(self.center)
        self.yaw_data.append(self.yaw)
        self.pitch_data.append(self.pitch)
        self.fwd_data.append(self.fwd)
        self.roll_data.append(self.roll)
        self.freq_offset_data.append(self.freq_offset)
        self.period_data.append(self.period)
        self.q_off_data.append(self.q_off)
        self.sw_data.append(self.sw)
        self.kpv_data.append(self.kpv)
        self.learn_data.append(self.learn)
        self.d_pinv_data.append(self.d_pinv)
        self.kp_th_data.append(self.kp_th)
        self.w_th_data.append(self.w_th)
        self.kp_s_data.append(self.kp_s)

    def save_data(self):
        np.savez(self.folder_name + "_np_data", q=self.q_data, dq=self.dq_data, t=self.timestamps, input=self.input_data, u=self.u_data, qd=self.qd_data, dqd=self.dqd_data)  

    def turtle_mode_callback(self,msg):

        self.mode = msg.mode
        self.traj_str = msg.traj

    def turtle_ctrl_callback(self, msg):
       

        # self.amplitude_data.append(msg.amplitude)
        # self.center_data.append(msg.center)
        # self.yaw_data.append(msg.yaw)
        # self.pitch_data.append(msg.pitch)
        # self.freq_offset_data.append(msg.frequency_offset)
        # self.period_data.append(msg.period)
        # self.q_off_data.append(msg.offset)
        # self.sw_data.append(msg.sw)
        # self.kpv_data.append(msg.kpv)
        # self.learn_data.append(msg.learned)
        # self.d_pinv_data.append(msg.d_pinv)
        # self.kp_th_data.append(msg.kp_th)
        # self.w_th_data.append(msg.w_th)
        # self.kp_s_data.append(msg.kp_s)

        self.amplitude = msg.amplitude
        self.center = msg.center
        self.yaw = msg.yaw
        self.pitch = msg.pitch
        self.fwd = msg.fwd
        self.roll = msg.roll
        self.freq_offset = msg.frequency_offset
        self.period = msg.period
        self.q_off = msg.offset
        self.sw = msg.sw
        self.kpv = msg.kpv
        self.learn = msg.learned
        self.d_pinv = msg.d_pinv
        self.kp_th = msg.kp_th
        self.w_th = msg.w_th
        self.kp_s = msg.kp_s
        # self.save_config()
        # self.save_data()
        self.reset()
        
    def save_config(self):  
        print(self.kpv_data)
        np.savez(self.folder_name + "_np_data", amplitude=self.amplitude_data, 
                 center=self.center_data, yaw=self.yaw_data, pitch=self.pitch_data, fwd=self.fwd_data, roll=self.roll_data, 
                 frequency_offset=self.freq_offset_data, period=self.period_data, offset=self.q_off_data,
                 sw=self.sw_data, kpv=self.kpv_data, learn=self.learn_data, d_pinv=self.d_pinv_data,
                 kp_th=self.kp_th_data, w_th=self.w_th_data, kps = self.kp_s_data
                 )
    
    def reset(self):
        self.q_data = []
        self.dq_data = []
        self.u_data = []
        self.input_data = []
        self.timestamps = []
        self.qd_data = []
        self.dqd_data = []

def main():
    rclpy.init()
    logger_node = Logger()
    try:
        rclpy.spin(logger_node)
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
    # print(logger_node.q_data)
    logger_node.save_data()
    logger_node.save_config()
    
        

if __name__ == '__main__':
    main()