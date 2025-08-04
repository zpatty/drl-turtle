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
from Gamepad import Gamepad as GP
import time

class XboxNew(GP.Gamepad):
    fullName = 'Xbox new'

    def __init__(self, joystickNumber = 0):
        GP.Gamepad.__init__(self, joystickNumber)
        self.axisNames = {
            0: 'LEFT-X',
            1: 'LEFT-Y',
            2: 'L2',
            3: 'RIGHT-X',
            4: 'RIGHT-Y',
            5: 'R2',
            6: 'DPAD-X',
            7: 'DPAD-Y'
        }
        self.buttonNames = {
            0:  'A',
            1:  'B',
            2:  'X',
            3:  'Y',
            4:  'LB',
            5:  'RB',
            6:  'HOME',
            7:  'START',
            8:  'XBOX',
            9:  'LASB',
            10: 'RASB',
        }
        self._setupReverseMaps()

class GamePad(Node):
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
        print("Game pad settings")
        # Gamepad settings
        gamepadType = XboxNew
        self.buttonRest = 'A'
        self.buttonTrack = 'Y'
        self.buttonAlt = 'X'
        self.buttonDR = 'B'
        self.buttonDD = 'RB'
        self.buttonDU = 'LB'
        self.buttonExit = 'HOME'
        self.toggleMode = 'START'
        self.joystickFwd = 'LEFT-Y'
        self.triggerLRoll = 'L2'
        self.triggerRRoll = 'R2'
        self.joystickTurn = 'RIGHT-X'
        self.joystickPitch = 'RIGHT-Y'
        self.pollInterval = 0.01

        # Wait for a connection
        if not GP.available():
            print('Please connect your gamepad...')
            while not GP.available():
                time.sleep(1.0)
        self.gamepad = gamepadType()
        print('Gamepad connected')

        # Set some initial state
        self.speed = 0.0
        
        self.pitch = 0.0
        self.auto = 0
        self.tracker = 0

        self.depth_d = 0.0
        self.altitude_d = 0.0
        self.yaw = 0.0

        # Start the background updating
        self.gamepad.startBackgroundUpdates()


        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.folder_name =  "data/" + t
        os.makedirs(self.folder_name)
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
                        print('Position')
                        mode_msg.mode = "p"
                        self.mode = mode_msg.mode
                        self.mode_pub.publish(mode_msg)
                    

                # Check for happy button changes
                if self.gamepad.beenPressed(self.toggleMode):
                    if self.auto:
                        self.auto = 0
                        self.tracker = 0
                        print("RC mode\n")
                  
                    else:
                        self.auto = 1
                        self.tracker = 0
                        print("Automous mode\n")
                
                if self.tracker != 1:
                    if self.gamepad.beenPressed(self.buttonAlt) and self.auto == 1:
                        self.auto = 2
                        print("Altitude Tracking\n")
                    elif self.auto == 1:
                        self.auto = 1
                        print("Depth Tracking\n")
                    
                    if self.gamepad.beenPressed(self.buttonTrack):
                        self.auto = 4
                        self.tracker = 1
                        print("Tracker\n")

                    if self.gamepad.beenPressed(self.buttonDR):
                        self.auto = 3
                        print("depth plus remote\n")
                else:
                    if self.gamepad.beenPressed(self.buttonAlt):
                        self.auto = 2
                    if self.gamepad.beenPressed(self.buttonDR):
                        self.auto = 3
                    if self.gamepad.beenPressed(self.buttonTrack):
                        self.auto = 0
                        self.tracker = 0

                
                if self.gamepad.beenPressed(self.buttonDD):
                    cfg_msg = TurtleCtrl()
                    self.depth_d += 0.2
                    cfg_msg.pitch = self.depth_d
                    self.config_pub.publish(cfg_msg)
                
                if self.gamepad.beenPressed(self.buttonDU):
                    cfg_msg = TurtleCtrl()
                    self.depth_d -= 0.2
                    cfg_msg.pitch = self.depth_d
                    self.config_pub.publish(cfg_msg)
                    

                # Check if the beep button is held
                if self.gamepad.isPressed(self.buttonExit):
                    mode_msg.mode = "kill"
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
        self.quat_data.append(msg.imu.quat.tolist())
        self.depth_sensor_data.append(msg.depth)
        self.alt_data.append(msg.altitude)
        self.depth_d_data.append(self.depth_d)
        self.alt_data.append(self.altitude)
        self.alt_d_data.append(self.altitude_d)
        self.yaw_data.append(self.yaw)

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

    def save_data(self):
        np.savez(self.folder_name + "_np_data", q=self.q_data, dq=self.dq_data, t=self.timestamps, input=self.input_data, 
                 u=self.u_data, qd=self.qd_data, dqd=self.dqd_data, depth=self.depth_sensor_data, depth_d=self.depth_d_data, 
                 quat = self.quat_data, alt=self.alt_data, alt_d=self.alt_d_data, yaw_d=self.yaw_data)  
            

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

if __name__ == '__main__':
    rclpy.init()
    remote_node = GamePad()
    try:
        rclpy.spin(remote_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[EXCEPTION] Some error occurred")
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    remote_node.gamepad.disconnect()
    remote_node.save_data()
    remote_node.save_config()
    remote_node.destroy_node()
    print("Saved data and config")
    rclpy.shutdown()
    print("Gamepad node shutdown")
