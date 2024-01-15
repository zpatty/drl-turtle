import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class                                  
from dyn_functions import *                    # Dynamixel support functions
from turtle_controller import *                # Controller 
from Constants import *                        # File of constant variables
from Mod import *
from utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from utilities import *
import json
import traceback
from queue import Queue
import serial

# global variable was set in the callback function directly
global mode
mode = 'rest'

class MinimalSubscriber(Node):

    def __init__(self, topic):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            topic,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard:' + msg.data)
        if msg.data == 'traj1':
            mode = 'traj1'

# TODO: perhaps abstract motor control further?
# class TurtleMotorsControl():
#     def __init__(self, portHandlerJoint, packetHandlerJoint, IDs):
#         if portHandlerJoint.openPort():
#             print("[MOTORS STATUS] Suceeded to open port")
#         else:
#             print("[ERROR] Failed to open port")
#         if portHandlerJoint.setBaudRate(BAUDRATE):
#             print("[MOTORS STATUS] Suceeded to open port")
#         else:
#             print("[ERROR] Failed to change baudrate")
#         IDs = [1,2,3,4,5,6]
#         self.nq = len(IDs)
#         self.Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
#         self.Joints.disable_torque()
#         self.Joints.set_current_cntrl_mode()
#         self.Joints.enable_torque()
#         self.q = np.array(self.Joints.get_position()).reshape(-1,1)
#         print(f"Our initial q: " + str(self.q))

class TurtleMotorsSubscriber(Node):

    def __init__(self, topic):
        super().__init__('minimal_subscriber')
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.listener_callback,
            10)
        self.motor_traj_sub = self.create_subscription(
            TurtleTraj,
            'motors_traj',
            self.trajectory_callback,
            10
        )
        self.cmd_received_pub = self.create_publisher(
            String,
            'turtle_state',
            10
        )
        self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(50)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.qds = np.zeros((6,1))
        self.dqds = np.zeros((6,1))
        self.ddqds = np.zeros((6,1))
        self.tvec = np.zeros((1,1))

    def listener_callback(self, msg):
        # continuously check battery and shut down motors and log error into logger 
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

    def stop_callback(self, msg):
        if msg.data == 'stop':
            self.emergency_stop = True
            self.mode = 'stop'
        elif msg.data == 'rest':
            self.emergency_stop = True
            self.mode = 'rest'
        # self.get_logger().info('I heard:' + self.mode)

def main(args=None):
    rclpy.init(args=args)
    threshold = 11.1
    motors_node = TurtleMotorsSubscriber('master_motors')
    try:
        xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        volt_string = xiao.readline()
        volts = float(volt_string[:-2])
        # log = "Battery Voltage: {volts}\n")
        log = "Battery Voltage: " + str(volts)
        print(log)
        # self.get_logger().info(log)
    except:
        print("uC not detected\n")

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
            volt_string = xiao.readline()
            volts = float(volt_string[:-2])
            if volts < threshold:
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
                    if volts < threshold:
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
                    if volts < threshold:
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
                volt_string = xiao.readline()
                volts = float(volt_string[:-2])
                if volts < threshold:
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
                    if volts < threshold:
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

