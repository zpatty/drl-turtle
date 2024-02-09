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

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import turtle_trajectory

ESC_ASCII_VALUE             = 0x1b
SPACE_ASCII_VALUE           = 0x20
WKEY_ASCII_VALUE            = 0x77
SKEY_ASCII_VALUE            = 0x73
AKEY_ASCII_VALUE            = 0x61
DKEY_ASCII_VALUE            = 0x64
CKEY_ASCII_VALUE            = 0x63
BKEY_ASCII_VALUE            = 0x62      # key to bend the top module
UKEY_ASCII_VALUE            = 0x75      # key to unbend the modules
NKEY_ASCII_VALUE            = 0x6E
IKEY_ASCII_VALUE            = 0x69 
PKEY_ASCII_VALUE            = 0x70  
QKEY_ASCII_VALUE            = 0x71 
TKEY_ASCII_VALUE            = 0x74   
RKEY_ASCII_VALUE            = 0x72     
MOD1_VALUE                  = 0x31      # pressing 1 on keyboard
MOD2_VALUE                  = 0x32
MOD3_VALUE                  = 0x33

import termios, fcntl, sys, os
from select import select
fd = sys.stdin.fileno()
old_term = termios.tcgetattr(fd)
new_term = termios.tcgetattr(fd)

global rest_received
global stop_received

rest_received = False
stop_received = False

global traj
traj = TurtleTraj()
def getch():
    new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
    termios.tcsetattr(fd, termios.TCSANOW, new_term)
    try:
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    return ch

def kbhit():
    new_term[3] = (new_term[3] & ~(termios.ICANON | termios.ECHO))
    termios.tcsetattr(fd, termios.TCSANOW, new_term)
    try:
        dr,dw,de = select([sys.stdin], [], [], 0)
        if dr != []:
            return 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        sys.stdout.flush()

    return 0
    
def turtle_state_callback(msg):
    global stop_received
    global rest_received
    print("MESSAGE RECEIVED")
    if msg.data == "rest_received":
        rest_received = True
        print("REST RECEIVED!!!!!!")
    elif msg.data == "stop_received":
        stop_received = True
        print("STOP RECEIVED!!!!!!")
    else:
        print("NOPE")

def save_data(acc_data=np.array([1,2,3]), gyr_data=np.array([1,2,3]), quat_data=np.array([1,2,3]), voltage_data=np.array([1,2,3]), q_data=np.array([1,2,3]), qd_data=np.array([1,2,3]), dq_data=np.array([1,2,3]), tau_data=np.array([1,2,3]), t_0=0, timestamps=np.array([1,2,3])):
    """
    Saves data of cyclical trajectory passed into the turtle 
    @param : q_data :  holds both q and dq arrays 
    """
    t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    folder_name =  "turtle_data/" + t
    os.makedirs(folder_name, exist_ok=True)
    scipy.io.savemat(folder_name + "/data.mat", {'acc_data': acc_data.T,'gyr_data': gyr_data.T,'quat_data': quat_data.T, 'voltage_data': voltage_data.T, 'q_data': q_data.T, 'dq_data': dq_data.T, 'qd_data': qd_data.T, 'tau_data': tau_data.T, 't_0': t_0, 'time_data': timestamps})

def turtle_data_callback(msg):
    """
    Callback function that opens turtle sensor data and saves it locally to machine
    """
    print("DATA CALLBACK CALLED\n")
    # extract and reopen numpy arrays
    quat_x = msg.imu.quat_x
    quat_y = msg.imu.quat_y
    quat_z = msg.imu.quat_z
    quat_w = msg.imu.quat_w

    quat_data = np.array([quat_x, quat_y, quat_z, quat_w])

    acc_x = msg.imu.acc_x
    acc_y = msg.imu.acc_y
    acc_z = msg.imu.acc_z

    acc_data = np.array([acc_x, acc_y, acc_z])

    gyr_x = msg.imu.gyr_x
    gyr_y = msg.imu.gyr_y
    gyr_z = msg.imu.gyr_z

    gyr_data = np.array([gyr_x, gyr_y, gyr_z])

    voltage_data = np.array(msg.voltage).reshape(1, len(msg.voltage))
    global traj
    t_0 = msg.t_0
    n = len(msg.timestamps)
    q_data = np.array(msg.q).reshape(10, n)
    dq_data = np.array(msg.dq).reshape(10, n)
    tau_data = np.array(msg.tau).reshape(10, n)
    len_qd = len(list(traj.tvec))
    qd_data = np.array(msg.qd).reshape(10, len_qd)
    dqd_data = np.array(msg.qd).reshape(10, len_qd)
    timestamps = np.array(msg.timestamps).reshape(1, n)

    save_data(acc_data=acc_data, gyr_data=gyr_data,quat_data=quat_data, 
              voltage_data=voltage_data, q_data=q_data, dq_data=dq_data, qd_data=qd_data,
                 tau_data=tau_data, t_0=t_0, timestamps=timestamps)
    # save_data(q_data=qwe_data, qd_data=qd_data,
    #              tau_data=tau_data, t_0=t_0, timestamps=timestamps)
    
    print("Data saved to folder!")

# tau_data=np.append(tau_data, tau_, axis=1) 
def main(args=None):
    def np2msg(mat):
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return list(squeezed)
    rclpy.init(args=args)
    global stop_received
    global rest_received
    global traj
    node = rclpy.create_node('keyboard_node')
    tomotors = node.create_publisher(String, 'turtle_mode_cmd', 10)
    traj_pub = node.create_publisher(TurtleTraj, 'turtle_traj', 10)
    turtle_sub = node.create_subscription(TurtleSensors, 'turtle_sensors', turtle_data_callback, 10)
    turtle_cmd_received = node.create_subscription(String, 'turtle_state', turtle_state_callback, 10)
    rate = node.create_rate(50)
    msg = String()
    traj = TurtleTraj()
    while rclpy.ok():
        print("\nT: Straight, W: Dive, B: TurnRF, C: TurnRR, R: Rest U: SURFACE, D: Custom Traj(or press SPACE to STOP!)")
        key_input = getch()
        if key_input == chr(SPACE_ASCII_VALUE):
            stop_received = False
            msg.data='stop'
            tomotors.publish(msg)
            while stop_received == False:
                rclpy.spin_once(node)
                msg.data='stop'
                tomotors.publish(msg)
                node.get_logger().info(msg.data)
            log = "Stop was sucessfully sent! Closing program...."
            node.get_logger().info(log)
            break           
            
        elif key_input == chr(WKEY_ASCII_VALUE) or key_input == chr(TKEY_ASCII_VALUE):    # print out the length changes
            if key_input == chr(TKEY_ASCII_VALUE):
                print("Sending STRAIGHT trajectory\n")
                # STRAIGHT
                qd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/straight/qd.mat', 'qd')
                dqd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/straight/dqd.mat', 'dqd')
                ddqd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/straight/ddqd.mat', 'ddqd')
                tvec = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/straight/tvec.mat', 'tvec')

                # print(f"tvec mat shape: {tvec.shape}\n")
                traj.qd = np2msg(qd_mat)
                traj.dqd = np2msg(dqd_mat)
                traj.ddqd = np2msg(ddqd_mat)
                # print(f"t vec list: {tvec.tolist()}")

                traj.tvec = tvec.tolist()[0]

                print("sent trajectories...")
                traj_pub.publish(traj)

            else:
                # msg.data='traj2'
                print("Sending DIVE trajectory\n")
                # DIVE 
                qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/dive/qd.mat', 'qd')
                dqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/dive/dqd.mat', 'dqd')
                ddqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/dive/ddqd.mat', 'ddqd')
                tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/dive/tvec.mat', 'tvec')

                # print(f"tvec mat shape: {tvec.shape}\n")
                traj.qd = np2msg(qd_mat)
                traj.dqd = np2msg(dqd_mat)
                traj.ddqd = np2msg(ddqd_mat)
                # print(f"t vec list: {tvec.tolist()}")

                traj.tvec = tvec.tolist()[0]

                print("sent trajectories...")
                traj_pub.publish(traj)
            # tomotors.publish(msg)
            # node.get_logger().info(msg.data)
        
        elif key_input == chr(CKEY_ASCII_VALUE):
            # TURNRR
            print("Sending TURNRR trajectory\n")
            qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrr/qd.mat', 'qd')
            dqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrr/dqd.mat', 'dqd')
            ddqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrr/ddqd.mat', 'ddqd')
            tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrr/tvec.mat', 'tvec')

            # print(f"tvec mat shape: {tvec.shape}\n")
            traj.qd = np2msg(qd_mat)
            traj.dqd = np2msg(dqd_mat)
            traj.ddqd = np2msg(ddqd_mat)
            # print(f"t vec list: {tvec.tolist()}")

            traj.tvec = tvec.tolist()[0]

            print("sent trajectories...")
            traj_pub.publish(traj)

        elif key_input == chr(BKEY_ASCII_VALUE):
            # TURNRF
            print("Sending TURNRF trajectory\n")
            qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrf/qd.mat', 'qd')
            dqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrf/dqd.mat', 'dqd')
            ddqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrf/ddqd.mat', 'ddqd')
            tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/turnrf/tvec.mat', 'tvec')

            # print(f"tvec mat shape: {tvec.shape}\n")
            traj.qd = np2msg(qd_mat)
            traj.dqd = np2msg(dqd_mat)
            traj.ddqd = np2msg(ddqd_mat)
            # print(f"t vec list: {tvec.tolist()}")

            traj.tvec = tvec.tolist()[0]

            print("sent trajectories...")
            traj_pub.publish(traj)
        elif key_input == chr(UKEY_ASCII_VALUE):
            # TURNRR
            print("Sending SURFACE trajectory\n")
            qd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/surface/qd.mat', 'qd')
            dqd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/surface/dqd.mat', 'dqd')
            ddqd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/surface/ddqd.mat', 'ddqd')
            tvec = mat2np('/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/surface/tvec.mat', 'tvec')

            # print(f"tvec mat shape: {tvec.shape}\n")
            traj.qd = np2msg(qd_mat)
            traj.dqd = np2msg(dqd_mat)
            traj.ddqd = np2msg(ddqd_mat)
            # print(f"t vec list: {tvec.tolist()}")

            traj.tvec = tvec.tolist()[0]

            print("sent trajectories...")
            traj_pub.publish(traj)

        elif key_input == chr(RKEY_ASCII_VALUE):
            msg.data='rest'
            tomotors.publish(msg)
            rest_received = False
            while rest_received == False:
                rclpy.spin_once(node)
                msg.data='rest'
                tomotors.publish(msg)
                node.get_logger().info(msg.data)
            log = "Rest command was sucessfully sent!"
            node.get_logger().info(log)
            
        elif key_input == chr(DKEY_ASCII_VALUE):
            qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/qd.mat', 'qd')
            dqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/dqd.mat', 'dqd')
            ddqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/ddqd.mat', 'ddqd')
            tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/tvec.mat', 'tvec')

            # print(f"tvec mat shape: {tvec.shape}\n")
            traj.qd = np2msg(qd_mat)
            traj.dqd = np2msg(dqd_mat)
            traj.ddqd = np2msg(ddqd_mat)
            # print(f"t vec list: {tvec.tolist()}")
            traj.tvec = tvec.tolist()[0]

            print("sent trajectories...")
            traj_pub.publish(traj)
        
        elif key_input == chr(PKEY_ASCII_VALUE):
            qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/qd_p.mat', 'qd')
            tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/tvec_p.mat', 'tvec')

            # print(f"tvec mat shape: {tvec.shape}\n")
            traj.qd = np2msg(qd_mat)
            traj.dqd = []
            traj.ddqd = []
            # print(f"t vec list: {tvec.tolist()}")
            traj.tvec = tvec.tolist()[0]
            print("sent trajectories...")
            traj_pub.publish(traj)
        else:
            print("Wrong input received")

    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass
