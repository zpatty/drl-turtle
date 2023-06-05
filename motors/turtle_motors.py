#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class                                  
from dyn_functions import *                    # Dynamixel support functions
from turtle_controller import *                # Controller 
# from logger import *                           # Data Logger
from Constants import *                        # File of constant variables
from Mod import *
from utilities import *
import json
import traceback
from queue import Queue
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    def kbhit():
        return msvcrt.kbhit()
else:
    import termios, fcntl, sys, os
    from select import select
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    new_term = termios.tcgetattr(fd)

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

# Open module port
# if portHandlerMod.openPort():
#     print("Succeeded to open the port")
# else:
#     print("Failed to open the port")
#     print("Press any key to terminate...")
#     getch()
#     quit()

# Set port baudrate
# if portHandlerMod.setBaudRate(BAUDRATE):
#     print("Succeeded to change the baudrate")
# else:
#     print("Failed to change the baudrate")
#     print("Press any key to terminate...")
#     getch()
#     quit()

# open big motors port
# Open joint port
if portHandlerJoint.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandlerJoint.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# packetHandlerMod = PacketHandler(PROTOCOL_VERSION)
packetHandlerJoint = PacketHandler(PROTOCOL_VERSION)

IDs = [4,5,6]
Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
Joints.set_current_cntrl_mode()
Joints.enable_torque()
t_old = time.time()

print("[DEBUG] dt:", (time.time() - t_old))  

nq = len(IDs)
th0 = 3.14/180 * np.array([180.0, 180.0, 360.0])#[180, 180, 270, 180, 180, 0]
# offset1 = joint_th0[1]
# offset1 = 3.745981084016736
# offset2 = joint_th0[2]
# offset2 = 4.414796707534875
# print(f"Offset for Motor Positions 1 and 2: {offset1, offset2}")
q = th0
q_old = q
# print(f"Motor Positions: {q}")
# our max arc length (in m)

print(f"Motor angles: {q}")
q_data = np.zeros((nq,1))
tau_data = np.zeros((nq,1))
timestamps = np.zeros((1,1))
dt_loop = np.zeros((1,1))       # hold dt data 

# Report our initial configuration
print(f"Our current q: {q}\n")
first_time = True
input_history = np.zeros((nq,5))

try: 
    while 1:
        print("\nP: Position Controlled Trajectory, C: Current Controlled Trajectory, W: Set point(or press SPACE to quit!)")
        key_input = getch()
        if key_input == chr(SPACE_ASCII_VALUE):
            print("we're quitting\n")
            break
        elif key_input == chr(PKEY_ASCII_VALUE):
            Joints.disable_torque()
            Joints.set_extended_pos_mode()
            Joints.enable_torque()
            print("\n1: Walk, 2: Swim (or press SPACE to quit!)")
            key_input = getch()
            if key_input == chr(SPACE_ASCII_VALUE):
                print("we're quitting\n")
                break
            
        elif key_input == chr(BKEY_ASCII_VALUE):
            # Back to home position 
            # put all motors in extended position mode
            # put them back into their reference motor step position
            Joints.disable_torque()
            Joints.set_extended_pos_mode()
            Joints.enable_torque()
            Joints.set_max_velocity(20)
            home = []
            for i in range(len(th0)):
                for j in range(len(th0[0])):
                    home.append(to_motor_steps(th0[i][j]))
            Joints.send_pos_cmd(home)
            break
        elif key_input == chr(WKEY_ASCII_VALUE):    # print out the length changes
            Joints.disable_torque()
            Joints.set_current_cntrl_mode()
            Joints.enable_torque()
            q_data = np.zeros((nq,1))
            tau_data = np.zeros((nq,1))
            timestamps = np.zeros((1,1))
            # Open up controller parameters
            # Kp, KD, config_params = parse_config()
            Kp = np.diag([1, 1, 1])*0.5
            KD = 0.07
            # Open up desired q from json file
            # qd = parse_setpoint(nq)
            qd = 3.14/180 * np.array([180.0, 180.0, 180.0]).reshape(-1,1)
            # qd = np.zeros((12,1)).reshape(-1,1)
            # print(f"[DEBUG] Our desired config: {qd}\n")
            # print(f"[DEBUG] Our error qd - q: {qd - q}")
            zero =  np.zeros((nq,1))
            t_old = time.time()
            # our loop's "starting" time
            t_0 = time.time()
            while 1:
                if kbhit():
                    c = getch()
                    first_time = True
                    Joints.send_torque_cmd(nq * [0])
                    print("[Q KEY PRESSED] : All motors stopped\n")
                    save_data(q_data, qd, tau_data, timestamps, config_params, dt_loop)
                    break
                else:

                    q = np.array(Joints.get_position()).reshape(-1,1)
                    # mj0 = joint_th[0] - base_offset
                    # mj1 = joint_th[1] - offset1
                    # mj2 = joint_th[2] - offset2

                    q_data=np.append(q_data, q, axis = 1) 
                    # At the first iteration velocity is 0  
                    if first_time:
                        dq = np.zeros((nq,1))
                        first_time = False
                    else:
                        t = time.time()
                        time_elapsed = t-t_0
                        timestamps = np.append(timestamps, time_elapsed) 
                        dt = t - t_old
                        # print(f"[DEBUG] dt: {dt}\n")  
                        t_old = t
                        dq = diff(q, q_old, dt)
    
                    # calculate errors
                    err = q - qd
                    err_dot = dq
                    # print(f"[DEBUG] q: {q}\n")
                    # TODO: set position limits
                    # tau = turtle_controller(q,dq,qd,zero,zero,Kp,KD)
                    # tau_data=np.append(tau_data, tau, axis=1) 
                    # print(f"[DEBUG] q: {q * 180/3.14}\n")
                    # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                    print(f"[DEBUG] e: {err * 180/3.14}\n")
                    q_old = q
                    # input = grab_arm_current(tau, min_torque, max_torque)
                    
                    tau = turtle_controller(q,dq,qd,zero,zero,Kp,KD)
                    
                    input_history = np.append(input_history[:,1:], tau,axis=1)

                    input_mean = np.mean(input_history, axis = 1)

                    input = grab_arm_current(input_mean, min_torque, max_torque)
                    
                    # print(f"[DEBUG] joint cmds: {tau}\n")
                    Joints.send_torque_cmd(input)
        elif key_input == chr(CKEY_ASCII_VALUE):    # Trajectory Code
            print(f"[MODE] TRAJECTORY\n")
            Joints.disable_torque()
            Joints.set_current_cntrl_mode()
            Joints.enable_torque()
            q_data = np.zeros((nq,1))
            tau_data = np.zeros((nq,1))
            timestamps = np.zeros((1,1))
            dt_loop = np.zeros((1,1))       # hold dt data 
            # Open up controller parameters
            # Kp, KD, config_params = parse_config()
            Kp = np.diag([0.5, 0.5, 0.5])*2
            KD = 0.5
            
            # Load desired trajectories from MATLAB
            qd_mat = mat2np('qd.mat', 'qd')
            dqd_mat = mat2np('dqd.mat', 'dqd')
            ddqd_mat = mat2np('ddqd.mat', 'ddqd')
            tvec = mat2np('tvec.mat', 'tvec')

            zero =  np.zeros((nq,1))
            t_old = time.time()
            # our loop's "starting" time
            t_0 = time.time()
            while 1:
                if kbhit():
                    c = getch()
                    first_time = True
                    Joints.send_torque_cmd(nq * [0])
                    print("[Q KEY PRESSED] : All motors stopped\n")
                    save_data(q_data, qd, tau_data, timestamps, config_params, dt_loop)

                    break
                else:
                    # grab current time   
                    # t1 = time.time()
                    q = np.array(Joints.get_position()).reshape(-1,1)
                    # t2 = time.time()
                    # print(f"[DEBUG] dt: {t2 - t1}\n")  
                    # mj0 = joint_th[0] - base_offset
                    # mj1 = joint_th[1] - offset1
                    # mj2 = joint_th[2] - offset2
                    # q = grab_arm_q(l[0], l[1], l[2], mj0, mj1, mj2, s, d)
                    # grab desired pos, vel and accel based off of time vector position 
                    
                    n = get_qindex((time.time() - t_0), tvec)
                    
                    qd = np.array(qd_mat[:, n]).reshape(-1,1)
                    dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                    ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                    # print(f"[DEBUG] qdata: {q_data}\n")
                    # print(f"[DEBUG] q: {q}\n")
                    q_data=np.append(q_data, q, axis = 1) 
                    # At the first iteration velocity is 0  
                    
                    if first_time:
                        dq = np.zeros((nq,1))
                        first_time = False
                    else:
                        t = time.time()
                        timestamps = np.append(timestamps, (t-t_0)) 
                        dt = t - t_old
                        print(f"[DEBUG] dt: {dt}\n")  
                        t_old = t
                        dq = diff(q, q_old, dt)
                        q_old = q
                    # calculate errors
                    err = q - qd
                    # print(f"[DEBUG] e: {err}\n")
                    # print(f"[DEBUG] q: {q * 180/3.14}\n")
                    # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                    err_dot = dq
                    
                    tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
                    
                    
                    input_history = np.append(input_history[:,1:], tau,axis=1)

                    input_mean = np.mean(input_history, axis = 1)

                    input = grab_arm_current(input_mean, min_torque, max_torque)

                    
                    
                    # print(f"[DEBUG] input: {input}\n")
                    Joints.send_torque_cmd(input)

                    
        elif key_input == chr(NKEY_ASCII_VALUE):
            # Update to new config
            m, l, config_params = parse_config()

    print("[END OF PROGRAM] Disabling torque\n")
    # Disable Dynamixel Torque
    Joints.disable_torque()
    # Close port
    # portHandlerMod.closePort()
except Exception:
    print("[ERROR] Disabling torque\n")
    Joints.disable_torque()
    traceback.print_exc()