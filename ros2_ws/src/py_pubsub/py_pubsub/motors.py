import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj
import sys
sys.path.append("/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub")
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
# [[[0.0,1.0,2.0,3.0,4.0,5.0], [6.0,7.0,8.0,9.0,10.0,11.0]], [[2.0,3.0,4.0,5.0,5.0,5.0], [3.0,4.0,5.0,9.0,10.0,11.0]], [[0.0,0.0,2.0,0.0,0.0,5.0], [6.0,0.0,0.0,9.0,0.0,0.0]], [1.0, 2.0]]

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
        # TODO: do an array of an array subscription
            # msg = [qd, dqd, ddqd, tvec]??
        # self.motor_pos_cmd = self.create_subscription(
        #     String,
        #     'motor_pos_cmd',
        #     self.stop_callback,
        #     10)
        self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(50)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.qds = np.zeros((6,1))
        self.dqds = np.zeros((6,1))
        self.ddqds = np.zeros((6,1))
        self.tvec = np.zeros((1,1))
        # # set up Dynamxiel ish
        # if portHandlerJoint.openPort():
        #     log = "Succeeded to open the port"
        # else:
        #     log = "Failed to open the port"
        # self.get_logger().info(log)
        # # Set port baudrate
        # if portHandlerJoint.setBaudRate(BAUDRATE):
        #     log = "Succeeded to change the baudrate"
        # else:
        #     log = "Failed to change the baudrate"
        # self.packetHandlerJoint = PacketHandler(PROTOCOL_VERSION)
        # self.get_logger().info(log)
        # # Instantiate the motors
        # # IDs = [1,2,3,4,5,6]
        # IDs = [1]
        # self.nq = len(IDs)
        # self.Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
        # self.Joints.disable_torque()
        # self.Joints.set_current_cntrl_mode()
        # self.Joints.enable_torque()
        # self.q = np.array(self.Joints.get_position()).reshape(-1,1)
        # log = "Our initial q: " + str(self.q)
        # self.get_logger().info(log)
        # self.volts = 15
        # try:
        #     self.xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        #     volt_string = self.xiao.readline()
        #     self.volts = float(volt_string[:-2])
        #     # log = "Battery Voltage: {volts}\n")
        #     log = "Battery Voltage: " + str(self.volts)
        #     self.get_logger().info(log)
        # except:
        #     log = "uC not detected\n"
        #     self.get_logger().info(log)
        # if self.volts < 11.5:
        #     log = "Time to charge, power off immediately\n"
        # self.get_logger().info(log)

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
        self.mode = 'traj_input'
        n = len(msg.tvec)
        self.qds = np.array(msg.qd).reshape(6,n)
        self.dqds = np.array(msg.dqd).reshape(6,n)
        self.ddqds = np.array(msg.ddqd).reshape(6,n)
        self.tvec = np.array(msg.tvec).reshape(1,n)
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
    motors_node = TurtleMotorsSubscriber('master_motors')
    # try:
    #     xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
    #     volt_string = xiao.readline()
    #     volts = float(volt_string[:-2])
    #     # log = "Battery Voltage: {volts}\n")
    #     log = "Battery Voltage: " + str(volts)
    #     print(log)
    #     # self.get_logger().info(log)
    # except:
    #     print("uC not detected\n")
        # self.get_logger().info(log)

    # set up dynamixel stuff
    # if portHandlerJoint.openPort():
    #     print("[MOTORS STATUS] Suceeded to open port")
    # else:
    #     print("[ERROR] Failed to open port")
    # if portHandlerJoint.setBaudRate(BAUDRATE):
    #     print("[MOTORS STATUS] Suceeded to open port")
    # else:
    #     print("[ERROR] Failed to change baudrate")
    IDs = [1,2,3,4,5,6]
    nq = len(IDs)
    # Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
    # Joints.disable_torque()
    # Joints.set_current_cntrl_mode()
    # Joints.enable_torque()
    # q = np.array(Joints.get_position()).reshape(-1,1)
    # print(f"Our initial q: " + str(q))
    print("going into while loop...")
    while rclpy.ok():
        # volt_string = xiao.readline()
        # volts = float(volt_string[:-2])
        # if volts < 11.5:
        #     Joints.disable_torque()
        #     break
        rclpy.spin_once(motors_node)
        if motors_node.mode == 'traj1':
            n = 5
            qd_mat = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/qd.mat', 'qd')
            dqd_mat = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/dqd.mat', 'dqd')
            ddqd_mat = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/ddqd.mat', 'ddqd')
            tvec = mat2np('/home/crush/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/tvec.mat', 'tvec')
            print(f"full thing is: {tvec.shape}\n")
            qd = np.array(qd_mat[:, n]).reshape(-1,1)
            # print(f"shape qd: {qd.shape}\n")
            # print(f"shape of tvec: {tvec.sh}")
            # Joints.send_torque_cmd(nq * [20])
            # first_time = True
            # input_history = np.zeros((nq,10))
            # q_data = np.zeros((nq,1))
            # tau_data = np.zeros((nq,1))
            # timestamps = np.zeros((1,1))
            # dt_loop = np.zeros((1,1))       # hold dt data 
            # first_time = True
            # print(f"[MODE] TRAJECTORY\n")
            # Joints.disable_torque()
            # Joints.set_current_cntrl_mode()
            # Joints.enable_torque()
            # q_data = np.zeros((nq,1))
            # tau_data = np.zeros((nq,1))
            # timestamps = np.zeros((1,1))
            # dt_loop = np.zeros((1,1))       # hold dt data 
            # Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*2
            # KD = 0.5
            # # Load desired trajectories from MATLAB
            # qd_mat = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/qd.mat', 'qd')
            # dqd_mat = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/dqd.mat', 'dqd')
            # ddqd_mat = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/ddqd.mat', 'ddqd')
            # tvec = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/tvec.mat', 'tvec')
            # # zero =  np.zeros((self.nq,1))
            # t_old = time.time()
            # # our loop's "starting" time
            # t_0 = time.time()
            # while 1:
            #     if volts < 11.5:
            #         Joints.disable_torque()
            #         break
            #     rclpy.spin_once(motors_node)
            #     if motors_node.mode == 'rest' or motors_node.mode == 'stop':
            #         Joints.send_torque_cmd(nq * [0])
            #         Joints.disable_torque()
            #         first_time = True
            #         break
                
            #     # TODO: test with all motors on turtle
            #     q = np.array(Joints.get_position()).reshape(-1,1)
                
            #     n = get_qindex((time.time() - t_0), tvec)
            #     if n == len(tvec[0])-1:
            #         t_0 = time.time() - tvec[0][200]
                
            #     if n == len(tvec[0]) - 1:
            #         t_0 = time.time()
                
            #     qd = np.array(qd_mat[:, n]).reshape(-1,1)
            #     print(f"shape qd: {qd.shape}\n")
            #     dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
            #     ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
            #     # # print(f"[DEBUG] qdata: {q_data}\n")
            #     # # print(f"[DEBUG] q: {q}\n")
            #     q_data=np.append(q_data, q, axis = 1) 
            #     # # At the first iteration velocity is 0  
                
            #     if first_time:
            #         dq = np.zeros((nq,1))
            #         q_old = q
            #         first_time = False
            #     else:
            #         t = time.time()
            #         timestamps = np.append(timestamps, (t-t_0)) 
            #         dt = t - t_old
            #     #     # print(f"[DEBUG] dt: {dt}\n")  
            #         t_old = t
            #         dq = diff(q, q_old, dt)
            #         q_old = q
            #     # # calculate errors
            #     err = q - qd
            #     # # print(f"[DEBUG] e: {err}\n")
            #     # # print(f"[DEBUG] q: {q * 180/3.14}\n")
            #     # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
            #     err_dot = dq
                
            #     tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
                
            #     input_history = np.append(input_history[:,1:], tau,axis=1)

            #     input_mean = np.mean(input_history, axis = 1)

            #     input = grab_arm_current(input_mean, min_torque, max_torque)
            #     # print(f"[DEBUG] tau: {tau}\n")
            #     # Joints.send_torque_cmd(input)
            # Joints.disable_torque()

        elif motors_node.mode == 'traj2':
            first_time = True
            # while 1:
            #     if volts < 11.5:
            #         Joints.disable_torque()
            #         break
            #     rclpy.spin_once(motors_node)
            #     if motors_node.mode == 'rest' or motors_node.mode == 'stop':
            #         Joints.send_torque_cmd(nq * [0])
            #         break
            #     Joints.send_torque_cmd(nq * [20])

            #     print("traj1 yall")
        elif motors_node.mode == 'stop':
            print("ending entire program...")
            print("disabling torques entirely...")
            # Joints.send_torque_cmd(nq * [0])
            # Joints.disable_torque()
            break
        elif motors_node.mode == 'rest':
            first_time = True
            # volt_string = xiao.readline()
            # volts = float(volt_string[:-2])
            # if volts < 11.5:
            #     Joints.disable_torque()
            #     break
            # print("rest mode....")
            # Joints.send_torque_cmd(nq * [0])
        elif motors_node.mode == 'traj_input':
            first_time = True
            # input_history = np.zeros((nq,10))
            # q_data = np.zeros((nq,1))
            # tau_data = np.zeros((nq,1))
            # timestamps = np.zeros((1,1))
            # dt_loop = np.zeros((1,1))       # hold dt data 
            # first_time = True
            # Joints.disable_torque()
            # Joints.set_current_cntrl_mode()
            # Joints.enable_torque()
            # q_data = np.zeros((nq,1))
            # tau_data = np.zeros((nq,1))
            # timestamps = np.zeros((1,1))
            # dt_loop = np.zeros((1,1))       # hold dt data 
            # Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*2
            # KD = 0.5
            # # Load desired trajectories from motors node
            qd_mat = motors_node.qds
            dqd_mat = motors_node.dqds
            ddqd_mat = motors_node.ddqds
            tvec = motors_node.tvec
            print(f"qd_mat.shape: {qd_mat.shape}")
            # print(f"dqd_mat.shape: {dqd_mat.shape}")
            # print(f"ddqd_mat.shape: {ddqd_mat.shape}")
            # print(f"tvec_mat.shape: {tvec.shape}")
            n = 0
            qd = np.array(qd_mat[:, n]).reshape(-1,1)
            print(f"qd shape: {qd.shape}")
            print(f"qd: {qd}")
            # # zero =  np.zeros((self.nq,1))
            # t_old = time.time()
            # # our loop's "starting" time
            # t_0 = time.time()
            # while 1:
            #     if volts < 11.5:
            #         Joints.disable_torque()
            #         break
            #     rclpy.spin_once(motors_node)

            #     if motors_node.mode == 'rest' or motors_node.mode == 'stop':
            #         Joints.send_torque_cmd(nq * [0])
            #         Joints.disable_torque()
            #         first_time = True
            #         break
                
            #     q = np.array(Joints.get_position()).reshape(-1,1)
            #     n = get_qindex((time.time() - t_0), tvec)
            #     if n == len(tvec[0])-1:
            #         t_0 = time.time() - tvec[0][200]
                
            #     if n == len(tvec[0]) - 1:
            #         t_0 = time.time()
                
            #     # TODO: add this from message
            #     qd = np.array(qd_mat[:, n]).reshape(-1,1)
            #     dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
            #     ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                # tvec = np.array(todo)
        else:
            print("wrong command received....")

    rclpy.shutdown()

if __name__ == '__main__':
    main()


#         elif msg.data == 'd2':
#             # first_time = True
#             # input_history = np.zeros((self.nq,10))
#             # q_data = np.zeros((self.nq,1))
#             # tau_data = np.zeros((self.nq,1))
#             # timestamps = np.zeros((1,1))
#             # dt_loop = np.zeros((1,1))       # hold dt data 
#             # first_time = True
#             # log = 'Running For loop Trajectory'
#             # self.get_logger().info(log)
#             # print(f"[MODE] TRAJECTORY\n")
#             # self.Joints.disable_torque()
#             # self.Joints.set_current_cntrl_mode()
#             # self.Joints.enable_torque()
#             # log = "structs"
#             # self.get_logger().info(log)
#             # q_data = np.zeros((self.nq,1))
#             # tau_data = np.zeros((self.nq,1))
#             # timestamps = np.zeros((1,1))
#             # dt_loop = np.zeros((1,1))       # hold dt data 
#             # Open up controller parameters
#             # Kp, KD, config_params = parse_config()
#             Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*2
#             KD = 0.5
#             # log = "uploading"
#             # self.get_logger().info(log)

#             # Load desired trajectories from MATLAB
#             qd_mat = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/qd.mat', 'qd')
#             dqd_mat = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/dqd.mat', 'dqd')
#             ddqd_mat = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/ddqd.mat', 'ddqd')
#             tvec = mat2np('/home/crush/drl-turtle/ros/py_pubsub/py_pubsub/tvec.mat', 'tvec')

#             # zero =  np.zeros((self.nq,1))
#             t_old = time.time()
#             # our loop's "starting" time
#             t_0 = time.time()
#             # log = "runnign loop"
#             # self.get_logger().info(log)
#             while 1:
#                 if msg.data == 'd3':
#                     # self.Joints.disable_torque()
#                     break
#                 log = "in the while loop"
#                 self.get_logger().info(log)
#                 # q = np.array(self.Joints.get_position()).reshape(-1,1)
#                  # q = np.array(self.Joints.get_position()).reshape(-1,1)
                
#                 # n = get_qindex((time.time() - t_0), tvec)
#                 # if n == len(tvec[0])-1:
#                 #     t_0 = time.time() - tvec[0][200]
                
#                 # if n == len(tvec[0]) - 1:
#                 #     t_0 = time.time()
                
#                 # qd = np.array(qd_mat[:, n]).reshape(-1,1)
#                 # dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
#                 # ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
#                 # q_data=np.append(q_data, q, axis = 1) 
#                 # # At the first iteration velocity is 0  
                
#                 # if first_time:
#                 #     dq = np.zeros((self.nq,1))
#                 #     q_old = q
#                 #     first_time = False
#                 # else:
#                 #     t = time.time()
#                 #     timestamps = np.append(timestamps, (t-t_0)) 
#                 #     dt = t - t_old
#                 #     t_old = t
#                 #     dq = diff(q, q_old, dt)
#                 #     q_old = q
#                 # # calculate errors
#                 # err = q - qd
#                 # err_dot = dq
                
#                 # tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
#                 # input_history = np.append(input_history[:,1:], tau,axis=1)
#                 # input_mean = np.mean(input_history, axis = 1)
#                 # inputs = grab_arm_current(input_mean, min_torque, max_torque)

#                 # self.Joints.send_torque_cmd(self.nq * [20])
#                 # n = get_qindex((time.time() - t_0), tvec)
#                 # if n == len(tvec[0])-1:
#                 #     t_0 = time.time() - tvec[0][200]
                
#                 # if n == len(tvec[0]) - 1:
#                 #     t_0 = time.time()
                
#                 # qd = np.array(qd_mat[:, n]).reshape(-1,1)
#                 # dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
#                 # ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
#                 # q_data=np.append(q_data, q, axis = 1) 
#                 # # At the first iteration velocity is 0  
                
#                 # if first_time:
#                 #     dq = np.zeros((self.nq,1))
#                 #     q_old = q
#                 #     first_time = False
#                 # else:
#                 #     t = time.time()
#                 #     timestamps = np.append(timestamps, (t-t_0)) 
#                 #     dt = t - t_old
#                                # err_dot = dq
#  #     t_old = t
#                 #     dq = diff(q, q_old, dt)
#                 #     q_old = q
#                 # # calculate errors
#                 # err = q - qd
#                 # err_dot = dq
                
#                 # tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
#                 # input_history = np.append(input_history[:,1:], tau,axis=1)
#                 # input_mean = np.mean(input_history, axis = 1)
#                 # inputs = grab_arm_current(input_mean, min_torque, max_torque)

#                 # self.Joints.send_torque_cmd(self.nq * [20])
#             # self.Joints.disable_torque()
#             log = "Ending Loop Trajectory"
#             self.get_logger().info(log)
#         if msg.data == 'd3':
#             # self.Joints.disable_torque()
#             log = 'Disabled motor torques'
#         elif msg.data == 'stop':
#             # self.Joints.disable_torque()
#             log = 'stopping'
#         else: 
#             log = 'I heard:' + msg.data            self.mode = 'rest'
