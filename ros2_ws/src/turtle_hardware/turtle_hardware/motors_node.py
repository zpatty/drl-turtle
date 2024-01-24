import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.turtle_controller import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from turtle_dynamixel.utilities import *
import json
import traceback
from queue import Queue
import serial

# global variable was set in the callback function directly
global mode
mode = 'rest'

class TurtleMotorsSubscriber(Node):
    """
    This motors node is responsible for continously reading voltage from sensors node and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    """

    def __init__(self, topic):
        super().__init__('motors_node')
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.turtle_mode_callback,
            10)
        # for case when trajectory mode, receives trajectory msg
        self.motor_traj_sub = self.create_subscription(
            TurtleTraj,
            'turtle_traj',
            self.trajectory_callback,
            10
        )
        # continously reads voltage from sensor node readout
        self.sensors_sub = self.create_subscription(
            TurtleSensors,
            'turtle_sensors',
            self.sensors_callback,
            10
        )
        # sends acknowledgement packet to keyboard node
        self.cmd_received_pub = self.create_publisher(
            String,
            'turtle_state',
            10
        )
        # continously publishes the current motor position      
        self.motor_pos_pub = self.create_publisher(
            String,
            'turtle_motor_pos',
            10
        )
        self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(50)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.qds = np.zeros((6,1))
        self.dqds = np.zeros((6,1))
        self.ddqds = np.zeros((6,1))
        self.tvec = np.zeros((1,1))
        self.voltage = 12.0

    def turtle_mode_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'traj1':
            self.mode = 'traj1'
        elif msg.data == 'traj2':
            self.mode = 'traj2'
        elif msg.data == 'stop':
            self.mode = 'stop'
        else:
            self.mode = 'rest'
    def trajectory_callback(self, msg):
        """
        Callback function that takes in list of squeezed arrays
        msg: [qd, dqd, ddqd, tvec]
        """    
        # print("traj callback\n")    
        n = len(msg.tvec)
        self.qds = np.array(msg.qd).reshape(6,n)
        self.dqds = np.array(msg.dqd).reshape(6,n)
        self.ddqds = np.array(msg.ddqd).reshape(6,n)
        self.tvec = np.array(msg.tvec).reshape(1,n)
        self.mode = 'traj_input'
    def sensors_callback(self, msg):
        """
        Callback function that takes in sensor data from turtle 
        """
        self.voltage = msg.voltage
        print(f"battery voltage: {self.voltage}\n")
    
def publish_motor_pos(q, node):
    """
    Method publishes motor positions to the turtle_motor_pos ros topic
    : param    q: array of motor positions read from dynamixels
    : param node: the TurtleMotorsSubscriber instance that is able to publish 
                  message to the turtle_motor_pos topic
    """
    pos_msg = TurtleMotorPos()
    pos_msg.q = q
    node.motor_pos_pub.publish(pos_msg)

def main(args=None):
    rclpy.init(args=args)
    threshold = 11.1
    motors_node = TurtleMotorsSubscriber('turtle_mode_cmd')

    # set up dynamixel stuff
    if portHandlerJoint.openPort():
        print("[MOTORS STATUS] Suceeded to open port")
    else:
        print("[ERROR] Failed to open port")
    if portHandlerJoint.setBaudRate(BAUDRATE):
        print("[MOTORS STATUS] Suceeded to open port")
    else:
        print("[ERROR] Failed to change baudrate")
    IDs = [1,2,3,4,5,6]
    nq = len(IDs)
    Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
    Joints.disable_torque()
    Joints.set_current_cntrl_mode()
    Joints.enable_torque()
    q = np.array(Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))
    print("going into while loop...")
    try: 
        while rclpy.ok():
            if motors_node.voltage < threshold:
                print("[WARNING] volt reading too low--closing program....")
                Joints.disable_torque()
                break
            rclpy.spin_once(motors_node)
            
            if motors_node.mode == 'traj1':
                # Load desired trajectories from MATLAB
                qd_mat = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/qd.mat', 'qd')
                dqd_mat = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/dqd.mat', 'dqd')
                ddqd_mat = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/ddqd.mat', 'ddqd')
                tvec = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/tvec.mat', 'tvec')
                

                print(f"full thing is: {tvec.shape}\n")
                print(f"shape qd_mat: {qd_mat.shape}\n")
                # print(f"shape of tvec: {tvec.sh}")
                first_time = True
                input_history = np.zeros((nq,10))
                q_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                timestamps = np.zeros((1,1))
                dt_loop = np.zeros((1,1))       # hold dt data 
                print(f"[MODE] TRAJECTORY\n")
                Joints.disable_torque()
                Joints.set_current_cntrl_mode()
                Joints.enable_torque()
                q_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                timestamps = np.zeros((1,1))
                dt_loop = np.zeros((1,1))       # hold dt data 
                Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*2
                KD = 0.5
                # zero =  np.zeros((self.nq,1))
                t_old = time.time()
                # our loop's "starting" time
                t_0 = time.time()
                while 1:
                    if motors_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        Joints.disable_torque()
                        break
                    rclpy.spin_once(motors_node)
                    # print("traj 1...")
                    if motors_node.mode == 'rest' or motors_node.mode == 'stop':
                        Joints.send_torque_cmd(nq * [0])
                        Joints.disable_torque()
                        first_time = True
                        break
                    
                    q = np.array(Joints.get_position()).reshape(-1,1)
                    
                    n = get_qindex((time.time() - t_0), tvec)
                    print(f"n: {n}\n")
                    if n == len(tvec[0])-1:
                        t_0 = time.time() - tvec[0][200]
                    
                    if n == len(tvec[0]) - 1:
                        t_0 = time.time()
                    
                    qd = np.array(qd_mat[:, n]).reshape(-1,1)
                    dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                    ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                    # # print(f"[DEBUG] qdata: {q_data}\n")
                    print(f"[DEBUG] qd: {qd}\n")
                    q_data=np.append(q_data, q, axis = 1) 
                    # # At the first iteration velocity is 0  
                    
                    if first_time:
                        dq = np.zeros((nq,1))
                        q_old = q
                        first_time = False
                    else:
                        t = time.time()
                        timestamps = np.append(timestamps, (t-t_0)) 
                        dt = t - t_old
                    #     # print(f"[DEBUG] dt: {dt}\n")  
                        t_old = t
                        dq = diff(q, q_old, dt)
                        q_old = q
                    # # calculate errors
                    err = q - qd
                    # # print(f"[DEBUG] e: {err}\n")
                    # # print(f"[DEBUG] q: {q * 180/3.14}\n")
                    # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                    err_dot = dq
                    
                    tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
                    
                    input_history = np.append(input_history[:,1:], tau,axis=1)

                    input_mean = np.mean(input_history, axis = 1)

                    input = grab_arm_current(input_mean, min_torque, max_torque)
                    # print(f"[DEBUG] tau: {tau}\n")
                    Joints.send_torque_cmd(input)
                Joints.disable_torque()

            elif motors_node.mode == 'traj2':
                first_time = True
                while 1:
                    if motors_node.voltage < threshold:
                        Joints.disable_torque()
                        break
                    rclpy.spin_once(motors_node)
                    if motors_node.mode == 'rest' or motors_node.mode == 'stop':
                        Joints.send_torque_cmd(nq * [0])
                        break
                #     Joints.send_torque_cmd(nq * [20])

                #     print("traj1 yall")
            elif motors_node.mode == 'stop':
                print("ending entire program...")
                print("disabling torques entirely...")
                Joints.send_torque_cmd(nq * [0])
                Joints.disable_torque()
                cmd_msg = String()
                cmd_msg.data = "stop_received"
                motors_node.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
                break
            elif motors_node.mode == 'rest':
                first_time = True
                if motors_node.voltage < threshold:
                    Joints.disable_torque()
                    print("THRESHOLD MET TURN OFFFF")
                    break
                print("rest mode....")
                Joints.send_torque_cmd(nq * [0])
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                motors_node.cmd_received_pub.publish(cmd_msg)
                print("sent rest received msg")
            elif motors_node.mode == 'traj_input':
                # Load desired trajectories from motors node
                print("traj input")
                qd_mat = motors_node.qds
                dqd_mat = motors_node.dqds
                ddqd_mat = motors_node.ddqds
                tvec = motors_node.tvec     
                # print(f"qd mat: {qd_mat.shape}\n")
                # print(f"qd mat first elemetn: {qd_mat[:, 1]}\n")
                print(f"full thing is: {tvec.shape}\n")
                print(f"shape qd_mat: {qd_mat.shape}\n")
                # print(f"shape of tvec: {tvec.sh}")
                first_time = True
                first_loop = True
                input_history = np.zeros((nq,10))
                q_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                timestamps = np.zeros((1,1))
                dt_loop = np.zeros((1,1))       # hold dt data 
                print(f"[MODE] TRAJECTORY\n")
                Joints.disable_torque()
                Joints.set_current_cntrl_mode()
                Joints.enable_torque()
                q_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                timestamps = np.zeros((1,1))
                dt_loop = np.zeros((1,1))       # hold dt data 
                Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*2
                KD = 0.5
                # zero =  np.zeros((self.nq,1))
                t_old = time.time()
                # our loop's "starting" time
                t_0 = time.time()
                while 1:
                    if motors_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        Joints.disable_torque()
                        break
                    rclpy.spin_once(motors_node)
                    # print("traj 1...")
                    if motors_node.mode == 'rest' or motors_node.mode == 'stop':
                        Joints.send_torque_cmd(nq * [0])
                        Joints.disable_torque()
                        first_time = True
                        break
                    
                    q = np.array(Joints.get_position()).reshape(-1,1)
                    if first_loop:
                        n = get_qindex((time.time() - t_0), tvec)
                    else:
                        # print("done with first loop")
                        offset = t_0 - 2
                        n = get_qindex((time.time() - offset), tvec)

                    # print(f"n: {n}\n")
                    if n == len(tvec[0]) - 1:
                        first_loop = False
                        t_0 = time.time()
                    
                    qd = np.array(qd_mat[:, n]).reshape(-1,1)
                    dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                    ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                    # # print(f"[DEBUG] qdata: {q_data}\n")
                    # print(f"[DEBUG] qd: {qd}\n")
                    q_data=np.append(q_data, q, axis = 1) 
                    # # At the first iteration velocity is 0  
                    
                    if first_time:
                        dq = np.zeros((nq,1))
                        q_old = q
                        first_time = False
                    else:
                        t = time.time()
                        timestamps = np.append(timestamps, (t-t_0)) 
                        dt = t - t_old
                    #     # print(f"[DEBUG] dt: {dt}\n")  
                        t_old = t
                        dq = diff(q, q_old, dt)
                        q_old = q
                    # # calculate errors
                    err = q - qd
                    # # print(f"[DEBUG] e: {err}\n")
                    # # print(f"[DEBUG] q: {q * 180/3.14}\n")
                    # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                    err_dot = dq
                    
                    tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
                    
                    input_history = np.append(input_history[:,1:], tau,axis=1)

                    input_mean = np.mean(input_history, axis = 1)

                    input = grab_arm_current(input_mean, min_torque, max_torque)
                    # print(f"[DEBUG] tau: {tau}\n")
                    Joints.send_torque_cmd(input)
                Joints.disable_torque()
            else:
                print("wrong command received....")
    except:
        Joints.send_torque_cmd(nq * [0])
        Joints.disable_torque()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

