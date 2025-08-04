from __future__ import print_function
import os
import sys
import json

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

import rclpy
from rclpy.node import Node
import numpy as np
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_interfaces.msg import TurtleState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray

from turtle_interfaces.msg import TurtleCtrl, TurtleMode

from datetime import datetime
# Load the gamepad and time libraries
import time


class Logger(Node):

    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    """

    def __init__(self, params=None):
        super().__init__('log_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )


        # timer_cb_group = None
        # self.call_timer = self.create_timer(0.01, self._config_cb, callback_group=timer_cb_group)

        # self.handler = ParameterEventHandler(self)

        # for case when trajectory mode, receives trajectory msg
        self.sensors_sub = self.create_subscription(
            TurtleState,
            'turtle_sensors',
            self.sensors_callback,
            qos_profile
        )

                # continously publishes the current motor position      
        self.desired_sub = self.create_subscription(
            TurtleCtrl,
            'turtle_ctrl_params',
            self.turtle_desired_callback,
            qos_profile
        )

        self.stereo_sub = self.create_subscription(
            Float32MultiArray,
            'stereo',
            self.stereo_callback,
            qos_profile
            )
        
        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.folder_name =  "data/" + t
        print(f"Attempting to create folder: {self.folder_name}", flush=True)
        os.makedirs(self.folder_name)
        print(f"Created folder: {self.folder_name}", flush=True)
        self.create_rate(100)
        params, __ = self.parse_ctrl_params()
        self.reset()
        self.mode = "rest"
        self.traj = "nav"

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
        self.quat_data.append(msg.imu.quat.tolist())
        self.depth_sensor_data.append(msg.depth)
        self.alt_data.append(msg.altitude)
        self.depth_d_data.append(self.depth_d)
        self.alt_data.append(self.altitude)
        self.alt_d_data.append(self.altitude_d)
        self.yaw_data.append(self.yaw)
        self.stereo_depth_data.append(self.stereo_depth)
        self.stereo_x_list.append(self.x)
        print(self.x)
        
        self.stereo_y_list.append(self.y)

        # control variables
        self.amplitude_data.append(self.amplitude)
        self.center_data.append(self.center)
        self.yaw_data.append(self.yaw)
        self.pitch_data.append(self.pitch)
        
        self.roll_data.append(self.roll)
        self.freq_offset_data.append(self.frequency_offset)
        self.period_data.append(self.period)
        self.q_off_data.append(self.offset)
        self.sw_data.append(self.sw)
        self.kpv_data.append(self.kpv)
        self.learn_data.append(self.learned)
        self.d_pinv_data.append(self.d_pinv)
        self.kp_th_data.append(self.kp_th)
        self.w_th_data.append(self.w_th)
        self.kp_s_data.append(self.kp_s)

    def turtle_desired_callback(self, msg):
        self.yaw = msg.yaw
        self.depth_d = msg.pitch
        self.altitude_d = msg.fwd

    def stereo_callback(self, msg):
        if msg.data[0] is not None:
            self.stereo_depth = msg.data[0]
        else:
            self.stereo_depth = np.nan

        self.x = msg.data[1]
        
        self.y = msg.data[2]
        # print(f"[DEBUG] \n x: ", self.x, "   y: ", self.y, "/n")

    def save_data(self):
        # print(np.shape(self.stereo_x_list))
        # print(np.shape(self.stereo_y_list))
        # print(np.shape(np.vstack((self.stereo_x_list, self.stereo_y_list))))
        print(np.shape(self.stereo_depth_data))
        np.savez(self.folder_name + "_np_data", q=self.q_data, dq=self.dq_data, t=self.timestamps, input=self.input_data, 
                 u=self.u_data, qd=self.qd_data, dqd=self.dqd_data, depth=self.depth_sensor_data, depth_d=self.depth_d_data, 
                 quat = self.quat_data, alt=self.alt_data, alt_d=self.alt_d_data, yaw_d=self.yaw_data, 
                 stereo_depth=self.stereo_depth_data, stereo_point=np.vstack((self.stereo_x_list, self.stereo_y_list)))
            

    def update_config(self):
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.folder_name =  "data/" + t
        self.save_data()
        self.save_config()
        self.reset()
        params, config_params = self.parse_ctrl_params()

        cfg_msg = TurtleCtrl()
        mode_msg = TurtleMode()
        print(params)
        # if params["mode"] == " " or params["mode"] == "":
        #     mode_msg.mode = "rest"
        # else:
        #     mode_msg.mode = params["mode"]

        # mode_msg.traj = params["traj"]

        cfg_msg.kp = params["kp"]
        # print("HJere")
        cfg_msg.kd = params["kd"]

        # these should be messages instead of parameters
        cfg_msg.amplitude = params["amplitude"]
        cfg_msg.center = params["center"]
        cfg_msg.yaw = params["yaw"]
        cfg_msg.pitch = params["depth"]
        cfg_msg.fwd = params["altitude"]
        cfg_msg.roll = params["roll"]
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
        # print(cfg_msg)
        # if mode_msg.traj != self.traj or mode_msg.mode != self.mode:
        #     self.mode_pub.publish(mode_msg)
        #     print("sent mode message")
        self.config_pub.publish(cfg_msg)
        # self.traj = mode_msg.traj
        # self.mode = mode_msg.mode
        
        self.get_logger().info('Updated Config')

    def parse_ctrl_params(self):
        with open('ctrl_config.json') as config:
            params = json.load(config)
            self._last_update = os.fstat(config.fileno()).st_mtime
        # print(f"[MESSAGE] Config: {param}\n")    
        # Serializing json
        config_params = json.dumps(params, indent=14)
        self.kp = params["kp"]
            # print("HJere")
        self.kd = params["kd"]

        # these should be messages instead of parameters
        self.amplitude = params["amplitude"]
        self.center = params["center"]
        self.yaw = params["yaw"]
        self.depth_d = params["depth"]
        self.altitude = params["altitude"]
        self.roll = params["roll"]
        self.frequency_offset = params["frequency_offset"]
        self.period = params["period"]



        self.kpv = params["kpv"]
        self.d_pinv = params["d_pinv"]
        self.learned = bool(params["learned"])
        self.kp_s = params["kp_s"]
        self.w_th = params["w_th"]
        self.kp_th = params["kp_th"]
        self.offset = params["offset"]
        self.sw = params["sw"]
        return params, config_params
    
    def save_config(self):  
        # print(self.kpv_data)
        np.savez(self.folder_name + "_np_config", amplitude=self.amplitude_data, 
                 center=self.center_data, yaw=self.yaw_data, depth=self.depth_d_data, altitude=self.alt_data, roll=self.roll_data, 
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
        self.depth_sensor_data = []
        self.quat_data = []
        self.alt_data = []
        self.alt_d_data = []
        self.yaw_data = []


        self.amplitude_data = []
        self.center_data = []
        self.pitch_data = []
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

        self.depth_d_data = []
        self.stereo_depth = np.nan
        self.x = 0.0
        self.y = 0.0
        self.altitude_d = []
        self.pitch = []
        self.stereo_depth_data = []
        self.stereo_x_list = []
        self.stereo_y_list = []

    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt

if __name__ == '__main__':
    rclpy.init()
    logger_node = Logger()
    try:
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("some error occurred")
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    logger_node.save_data()
    logger_node.save_config()
    print("Saved data and config")
    logger_node.destroy_node()
    rclpy.shutdown()
    print("Logger node shutdown")
    sys.exit(0)
