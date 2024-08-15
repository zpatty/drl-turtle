import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import sys
import os
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

from turtle_interfaces.msg import TurtleTraj, TurtleSensors
import transforms3d.quaternions as quat
# import torch
# from EPHE import EPHE
# from DualCPG import DualCPG
# from AukeCPG import AukeCPG
from continuous_primitive import *


# import gymnasium as gym
# from gymnasium import spaces
import re
import cv2



from std_msgs.msg import String
from dynamixel_sdk import *                                     # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.turtle_controller import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
from turtle_dynamixel.utilities import *
import numpy as np
import json
import serial
import random
import traceback
from TurtleRobot import TurtleRobot

from statistics import mode as md
# print(f"sec import toc: {time.time()-tic}")

# tic = time.time()
os.system('sudo /home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel/latency_write.sh')

# global variable was set in the callback function directly
# global mode
# mode = 'rest'


def execute_data_primitive(turtle_node, primitive, t_0):
    if primitive != "dwell":
        print(f"---------------------------------------PRIMITIVE: {primitive}\n\n")
        folder = '/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/'
        qd_mat = mat2np(f'{folder + primitive}/qd.mat', 'qd')
        dqd_mat = mat2np(f'{folder + primitive}/dqd.mat', 'dqd')
        ddqd_mat = mat2np(f'{folder + primitive}/ddqd.mat', 'ddqd')
        tvec = mat2np(f'{folder + primitive}/tvec.mat', 'tvec')
        first_time = True
        first_loop = True
        input_history = np.zeros((turtle_node.nq,10))
        q_data = np.zeros((turtle_node.nq,1))
        tau_data = np.zeros((turtle_node.nq,1))
        timestamps = np.zeros((1,1))
        dt_loop = np.zeros((1,1))       # hold dt data 
        dq_data = np.zeros((turtle_node.nq,1))
        tau_data = np.zeros((turtle_node.nq,1))
        timestamps = np.zeros((1,1))
        dt_loop = np.zeros((1,1))       # hold dt data 
        cycle = 0


        # our loop's "starting" time
        # t_0 = time.time()
        t = time.time() - t_0
                        
        # if first_loop:
        n = get_qindex(t, tvec)

        if n > len(tvec[0]) - 2:
            first_loop = False
            t_0 = time.time()
            cycle += 1
            print(f"-----------------cycle: {cycle}\n\n\n")
        
        qd = np.array(qd_mat[:, n]).reshape(-1,1)
        dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
        ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
        # # print(f"[DEBUG] qdata: {q_data}\n")
        # print(f"[DEBUG] qd: {qd}\n")
        q, dq = turtle_node.get_state()

        # # calculate errors
        err = q - qd
        # # print(f"[DEBUG] e: {err}\n")
        # # print(f"[DEBUG] q: {q * 180/3.14}\n")
        # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
        err_dot = dq

        tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp*5,turtle_node.KD)

        # publish motor data, tau data, 
        
        input_history = np.append(input_history[:,1:], tau,axis=1)
        input_mean = np.mean(input_history, axis = 1)
        curr = grab_arm_current(input_mean, min_torque, max_torque)
        turtle_node.Joints.send_torque_cmd(curr)
        turtle_node.log_time(t)
        turtle_node.log_u(tau)
    else:
        t = time.time() - t_0
        tau = [0]*10
        turtle_node.log_time(t)
        turtle_node.log_u(tau)
        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))

    return t_0

def execute_primitive_pos(turtle_node, primitive, t_0):
    
    input_history = np.zeros((turtle_node.nq,10))
    q_data = np.zeros((turtle_node.nq,1))
    dq_data = np.zeros((turtle_node.nq,1))
    
    if turtle_node.voltage < turtle_node.voltage_threshold:
        print("voltage too low--powering off...")
        turtle_node.Joints.disable_torque()
        return
    rclpy.spin_once(turtle_node)
    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
        turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
        turtle_node.Joints.disable_torque()
        first_time = True
        return

    # check for new trajectory information
    
    q, dq = turtle_node.get_state()
    # if first_loop:]
    q_prim, _ = convert_motors_to_q(q, q)

    # if turtle_node.ctrl_flag:
    t = time.time() - t_0
    
    w = 2 * np.pi / turtle_node.period
    qd, dqd, ddqd = primitive(t, q_prim, w, turtle_node.amplitude, turtle_node.center, turtle_node.yaw, turtle_node.freq_offset, turtle_node.pitch)
    qd, dqd, ddqd = convert_q_to_motors(qd, dqd, ddqd)
    # # print(f"[DEBUG] qdata: {q_data}\n")
    # print(f"[DEBUG] qd: {qd * 180 / np.pi}\n")
    
    # # At the first iteration velocity is 0  
    
    # # calculate errors
    err = q - qd
    # # print(f"[DEBUG] e: {err}\n")
    # # print(f"[DEBUG] q: {q * 180/3.14}\n")
    # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
    err_dot = dq
    tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp * 5,turtle_node.KD)
    # publish motor data, tau data, 
    
    input_history = np.append(input_history[:,1:], tau,axis=1)
    input_mean = np.mean(input_history, axis = 1)
    curr = grab_arm_current(input_mean, min_torque, max_torque)
    turtle_node.Joints.send_torque_cmd(curr)
    turtle_node.log_time(t)
    turtle_node.log_u(tau)
    # print("here")
    
    return t_0

def execute_primitive_vel(turtle_node, primitive, t_0):
    input_history = np.zeros((turtle_node.nq,10))
    # q_data = np.zeros((turtle_node.nq,1))
    # dq_data = np.zeros((turtle_node.nq,1))
    
    # check for new trajectory information
    
    q, dq = turtle_node.get_state()
    # turtle_node.q_data.append(q, axis = 1)
    # if first_loop:
    q_prim, _ = convert_motors_to_q(np.array(q), np.array(q))
    t = time.time() - t_0
    u, aux = primitive(t, q_prim[:6])
    
    _, ud, _ = convert_q_to_motors(u, u, u)

    # print(np.array(q[6:]) - np.ones((1,4)) * np.pi)
    ud[6:] = -np.array(q[6:]).reshape(4,1) + np.ones((4,1)) * np.pi
    # # print(f"[DEBUG] qdata: {q_data}\n")
    # print(f"[DEBUG] qd: {qd}\n")
    
    # # At the first iteration velocity is 0  
    
    # # calculate errors
    # err = q - qd
    # # print(f"[DEBUG] e: {err}\n")
    # # print(f"[DEBUG] q: {q * 180/3.14}\n")
    # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
    # err_dot = dq

    # tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)

    # publish motor data, tau data, 
    
    # curr = grab_arm_current(input_mean, min_torque, max_torque)
    # print(ud)
    vd_ticks = np.clip(ud * 30.0 / np.pi / 0.229, -160., 160.).astype(int).squeeze().tolist()

    # print((ud * 30.0 / np.pi / 0.229).astype(int).squeeze().tolist())
    turtle_node.Joints.send_vel_cmd(vd_ticks)
    turtle_node.log_time(t)
    turtle_node.log_u(ud)
    return t_0

def execute_primitive_learned(turtle_node, primitive, t_0):
    input_history = np.zeros((turtle_node.nq,10))
    # q_data = np.zeros((turtle_node.nq,1))
    # dq_data = np.zeros((turtle_node.nq,1))
    
    # check for new trajectory information
    
    q, dq = turtle_node.get_state()
    turtle_node.q_data.append(q, axis = 1)
    # if first_loop:
    q_prim, _ = convert_motors_to_q(q, q)
    t = time.time() - t_0
    u = turtle_node.ud
    
    _, ud, _ = convert_q_to_motors(u, u, u)
    # # print(f"[DEBUG] qdata: {q_data}\n")
    # print(f"[DEBUG] qd: {qd}\n")
    
    # # At the first iteration velocity is 0  
    
    # # calculate errors
    # err = q - qd
    # # print(f"[DEBUG] e: {err}\n")
    # # print(f"[DEBUG] q: {q * 180/3.14}\n")
    # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
    # err_dot = dq

    # tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)

    # publish motor data, tau data, 
    
    # curr = grab_arm_current(input_mean, min_torque, max_torque)
    # print(ud)
    vd_ticks = np.clip(ud * 30.0 / np.pi / 0.229, -160., 160.).astype(int).squeeze().tolist()

    # print((ud * 30.0 / np.pi / 0.229).astype(int).squeeze().tolist())
    turtle_node.Joints.send_vel_cmd(vd_ticks)
    turtle_node.log_time(t)
    turtle_node.log_u(ud)
    return t_0

def get_primitive_and_execution(turtle_node, first_time):
    match turtle_node.get_turtle_mode():
        case 'planner':
            """
            Take commands from a planner and run data based primitives
            """
            if first_time:
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()
            try:
                primitive = md(turtle_node.primitives)
            except:
                primitive = "dwell"
            turtle_node.primitives = []
            execution = execute_data_primitive
            
        case 'cont':
            """
            Take commands from a planner and run continous primitives
            """
            if first_time:
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()

            primitive = sinusoidal_primitive
            execution = execute_primitive_pos

        case 'jsms':
            """
            Randomly pick a motion primitive and run it 4-5 times
            """
            if first_time:
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_velocity_cntrl_mode()
                turtle_node.Joints.enable_torque()
            primitive = joint_space_control_fn

            execution = execute_primitive_vel

        case 'tsms':
            """
            Randomly pick a motion primitive and run it 4-5 times
            """
            if first_time:
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_velocity_cntrl_mode()
                turtle_node.Joints.enable_torque()
            primitive = task_space_control_fn

            execution = execute_primitive_vel

        case 'ltsms':
            """
            Randomly pick a motion primitive and run it 4-5 times
            """
            if first_time:
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_velocity_cntrl_mode()
                turtle_node.Joints.enable_torque()
            primitive = task_space_control_fn_learned

            execution = execute_primitive_vel

        case 'comp':
            if first_time:
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_velocity_cntrl_mode()
                turtle_node.Joints.enable_torque()
            
            primitive = 'learned'
            execution = execute_primitive_learned
        
        case 'rest':
            primitive = 'rest'
            execution = execute_data_primitive
        
        case 'stop':
            primitive = 'rest'
            execution = execute_data_primitive

        case _:
            primitive = 'rest'
            execution = execute_data_primitive
        

    if turtle_node.get_turtle_mode() in turtle_node.turtle_trajs:
        if first_time:
            turtle_node.Joints.disable_torque()
            turtle_node.Joints.set_current_cntrl_mode()
            turtle_node.Joints.enable_torque()
        primitive = turtle_node.get_turtle_mode()
        execution = execute_data_primitive

    return primitive, execution

def main(args=None):
    rclpy.init(args=args)
    
    turtle_node = TurtleRobot('turtle_mode_cmd')
    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))
    


    # create folders 
    try:
        typed_name =  input("give folder a name: ")
        turtle_node.folder_name = "data/" + typed_name
        os.makedirs(turtle_node.folder_name, exist_ok=False)

    except:
        print("received weird input or folder already exists-- naming folder with timestamp")
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        turtle_node.folder_name =  "data/" + t
        os.makedirs(turtle_node.folder_name)

    
    last_mode = ''
    best_reward = 0
    try: 
        while rclpy.ok():
            rclpy.spin_once(turtle_node)
            mode = turtle_node.get_turtle_mode()
            if mode != last_mode:
                print(f"turtle mode: {mode}\n")
            last_mode = mode
            try:

                # turtle_node.read_voltage()
                if turtle_node.check_end():
                    break
                first_time = True
                t_0 = time.time()
                t_last = t_0
                mode = turtle_node.get_turtle_mode()
                while not turtle_node.check_end() and turtle_node.get_turtle_mode() != 'rest':
                    
                    rclpy.spin_once(turtle_node)
                    # turtle_node.read_voltage()
                    primitive, execution = get_primitive_and_execution(turtle_node, first_time)
                    t = time.time()
                    print(f"[DEBUG] dt: {t - t_last}\n")
                    t_last = t
                    if primitive == 'rest':
                        break
                    first_time = False
                    turtle_node.read_joints()
                    turtle_node.log_joints()
                    
                    t_0 = execution(turtle_node, primitive, t_0)
                    # turtle_node.publish_turtle_data()
                    
                
                # turtle_node.publish_turtle_data()
                turtle_node.shutdown_motors()

                # check for mode to enable proper control protocol only on first time
                # read and publish sensors
                # execute next command in the protocol
                
            except:
                print(traceback.format_exc())
                pass
    except Exception as e:
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    turtle_node.shutdown_motors()
    print("Shutting Down")
    rclpy.shutdown()

if __name__ == '__main__':
    main()


