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


# Load the gamepad and time libraries
from Gamepad import Gamepad
import time


class GamePad(Node):

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
        self.call_timer = self.create_timer(0.01, self._config_cb, callback_group=timer_cb_group)

        # self.handler = ParameterEventHandler(self)


        # continously publishes the current motor position      
        self.config_pub = self.create_publisher(
            TurtleCtrl,
            'turtle_ctrl_params',
            qos_profile
        )

        # continously publishes the current motor position      
        self.gamepad_pub = self.create_publisher(
            Float32MultiArray,
            'gamepad',
            qos_profile
        )

        # continously publishes the current motor position      
        self.mode_pub = self.create_publisher(
            TurtleMode,
            'turtle_mode',
            qos_profile
        )

        # for case when trajectory mode, receives trajectory msg
        self.sensors_sub = self.create_subscription(
            TurtleState,
            'turtle_sensors',
            self.sensors_callback,
            qos_profile
        )
        # self.mode_cmd_sub       # prevent unused variable warning
        # Gamepad settings
        gamepadType = Gamepad.XboxNew
        self.buttonRest = 'A'
        self.buttonExit = 'HOME'
        self.toggleMode = 'START'
        self.joystickFwd = 'LEFT-Y'
        self.triggerLRoll = 'L2'
        self.triggerRRoll = 'R2'
        self.joystickTurn = 'RIGHT-X'
        self.joystickPitch = 'RIGHT-Y'
        self.pollInterval = 0.01

        # Wait for a connection
        if not Gamepad.available():
            print('Please connect your gamepad...')
            while not Gamepad.available():
                time.sleep(1.0)
        self.gamepad = gamepadType()
        print('Gamepad connected')

        # Set some initial state
        self.speed = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.auto = 0

        # Start the background updating
        self.gamepad.startBackgroundUpdates()


        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.folder_name =  "data/" + t
        # os.makedirs(self.folder_name)
        self.create_rate(100)
        params, __ = self.parse_ctrl_params()

        self.reset()

        self.mode = "rest"
        self.traj = "nav"



    
    def _config_cb(self):
        if self._last_update != os.stat('ctrl_config.json').st_mtime:
            self.update_config()
        try: 
            if self.gamepad.isConnected():
                cfg_msg = Float32MultiArray()
                mode_msg = TurtleMode()
                if self.gamepad.beenPressed(self.buttonRest):
                    if self.mode != "rest":
                        print('REST')
                        mode_msg.mode = "rest"
                        self.mode = mode_msg.mode
                        self.mode_pub.publish(mode_msg)
                    else:
                        print('Velocity')
                        mode_msg.mode = "v"
                        self.mode = mode_msg.mode
                        self.mode_pub.publish(mode_msg)
                    

                # Check for happy button changes
                if self.gamepad.beenPressed(self.toggleMode):
                    if self.auto:
                        self.auto = 0
                        print("RC mode\n")
                        
                    else:
                        self.auto = 1
                        print("Automous mode\n")
                        

                # Check if the beep button is held
                if self.gamepad.isPressed(self.buttonExit):
                    mode_msg.mode = "rest"
                    self.mode = mode_msg.mode
                    self.mode_pub.publish(mode_msg)
                    raise KeyboardInterrupt

                # Update the joystick positions
                # Speed control (inverted)
                fwd = -self.gamepad.axis(self.joystickFwd)
                # Steering control (not inverted)
                yaw = self.gamepad.axis(self.joystickTurn)
                pitch = self.gamepad.axis(self.joystickPitch)
                roll_left = self.gamepad.axis(self.triggerLRoll)
                roll_right = self.gamepad.axis(self.triggerRRoll)
                roll = - 0.5 * roll_left + 0.5 * roll_right
                self.gamepad_pub.publish(Float32MultiArray(data=[fwd, pitch, yaw, roll, self.auto]))
                # print('%+.1f %% speed, %+.1f %% yaw, %+.1f %% pitch' % (fwd * 100, yaw * 100, pitch * 100))
        except:
            print("no controller")
            raise KeyboardInterrupt
            # t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            # with open("data/" + t + "_config.json", 'w') as config:
            #     json.dump(params, config, ensure_ascii=False, indent=4)

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

    def save_data(self):
        np.savez(self.folder_name + "_np_data", q=self.q_data, dq=self.dq_data, t=self.timestamps, input=self.input_data, u=self.u_data, qd=self.qd_data, dqd=self.dqd_data)  
            

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
        cfg_msg.fwd = params["fwd"]
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
        if mode_msg.traj != self.traj or mode_msg.mode != self.mode:
            self.mode_pub.publish(mode_msg)
            print("sent mode message")
        self.config_pub.publish(cfg_msg)
        self.traj = mode_msg.traj
        self.mode = mode_msg.mode
        
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
        self.pitch = params["pitch"]
        self.fwd = params["fwd"]
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

if __name__ == '__main__':
    rclpy.init()
    remote_node = GamePad()
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
    remote_node.gamepad.disconnect()
    remote_node.save_data()
    remote_node.save_config()
