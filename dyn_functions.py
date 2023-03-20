#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import time
from datetime import datetime
from math import cos, sin
import numpy as np
import scipy.io
import pandas as pd
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class
from Constants import *
from logger import *
def grab_motor_lengths(x):
    """
    input: x (state variables) and current cable lengths
    grabs current configuration (x) and outputs the desired motor lengths we need 
    to input into the dynamixels
    """
    u1 = x[0][0]
    v1 = x[1][0]
    u2 = x[2][0]
    v2 = x[3][0]
    h1 = x[4][0]
    h2 = x[5][0]
    theta = x[6][0]
    arm_lens = [[0,0,0], [0,0,0]]

    u = [u1, u2]
    v = [v1, v2]
    h = [h1, h2]
    mod = [l1, l2]
    
    for j in range(len(u)):
        arm_lens[j][0] = h[j] - d*v[j] 
        arm_lens[j][1] = h[j]+ 0.5 * d * (v[j] + math.sqrt(3) * u[j])
        arm_lens[j][2] = h[j]+ 0.5 * d * (v[j] - math.sqrt(3) * u[j])
    # for each module
    for i in range(len(u)):
        # for each motor in module
        for j in range(len(arm_lens[0])):
            if (arm_lens[i][j] > upper_limit[i][j]):
                arm_lens[i][j] = upper_limit[i][j]
            elif (arm_lens[i][j] < lower_limit[i][j]):
                arm_lens[i][j] = lower_limit[i][j]
            else:
                continue
    
    return arm_lens
def calc_error(p_d, p_0):
    err = math.sqrt((p_d[0]-p_0[0])**2 + (p_d[1]-p_0[1])**2 + (p_d[2]-p_0[2])**2)
    return err
def to_motor_steps(th0):
    """
    Input: the theta needed (in radians)
    Returns: new motor step command to be inputted into set_goal_position method
    """
    steps = th0 * (4096/(2 *math.pi))
    return int(steps)
def to_radians(steps):
    """
    Input: takes in dynamixel motor steps pos
    Output: radian value of dynamixel
    """
    rad_val = (steps * ((2 * math.pi)/4096))
    return rad_val
def to_rad_secs(revs):
    """
    Input:  Present velocity of dynamixel in rev/min
    Output: Velocity in rad/sec
    """
    rad_sec = revs * (2 * math.pi)/60
    return rad_sec
def diff(q, q_old, dt):
    """
    Input: current config and old config
    Output: calculates the derivative over some dt
    """
    q_dot = (q - q_old)/dt
    return q_dot

def get_time():
    """
    Output: returns time object you can use to calculate dt
    To read the time stamp, you do "t_obj.time()"
    """
    t = datetime.now().strftime("%H:%M:%S.%f")
    t_obj = datetime.strptime(t, "%H:%M:%S.%f")
    return t_obj

def get_qindex(mod_clock, tvec):
    """
    input: takes in the current time we're getting and makes sure we stay at proper
    time element 
    output: outputs the proper time vector index we need for our pos, vel, and accel vectors
    Notes:
    - recall tvec is a numpy matrix object that is (1,56) dim 
    """
    qindex = 0
    for t in range(1, tvec.shape[1]):
        if mod_clock >= tvec[0, t-1] and mod_clock < tvec[0, t]:
            qindex = t - 1
            return qindex
        elif t==(tvec.shape[1] - 1) and mod_clock >= tvec[0, t]:
            qindex = t
            return qindex
    return qindex

def mat2np(fname, typ):
    """
    Function that converts mat file to numpy matrix
    Parameters
    ----------------------------
        **both inputs are in string format
        fname = name of the file, i.e 'q.mat'
        typ = name of matrix you're trying to pull from dictionary, i.e 'q'
    """
    mat = scipy.io.loadmat(fname)[typ]
    # # for the 56 samples you created grab the gen coords
    return mat

def grab_current(tau_cables, min_torque, max_torque):
    """
    Parameters
    ----------------------------
    tau_cables: the raw currents calculated from controller (in mA?)
    min_torque: the minimum torque required to actuate module
    max_torque: the maximum torque module can be actuated without 
                damaging the module (basially point where plates touch)
    l: current cable lengths
        can do something like if current makes no
    Returns
    -----------------------------
    Ensures we only pass safe currents that the module can physically handle
    """
    mod_input = [0, 0, 0]
    for i in range(len(tau_cables)):
        if tau_cables[i][0].item() < 0:
            # negative current case
            if tau_cables[i][0].item() < -max_torque:
                mod_input[i] = -max_torque
            elif tau_cables[i][0].item() > -min_torque:
                mod_input[i] = 0
            else:
                mod_input[i] = int(tau_cables[i][0].item())
        else:    
            # positive current case
            if tau_cables[i][0].item() > max_torque:
                mod_input[i] = max_torque
            elif tau_cables[i][0].item() < min_torque:
                mod_input[i] = 0
            else:
                mod_input[i] = int(tau_cables[i][0].item())   
    m1 = mod_input[0]
    m2 = mod_input[1]
    m3 = mod_input[2]
    mod_input = [m1, m2, m3]
    # print(f"Our new mod input: {mod_input}\n")
    return mod_input

def grab_cable_lens(Mod1, l1, l1_0, th1_0, r):
    """
    Reads current motor angles from a module to calculate the current
    cable lengths.
    """
    dtheta1 = [0, 0, 0]
    for i in range(len(Mod1)):
        th1 = to_radians(Mod1[i].get_present_pos())
        dtheta1[i] = th1 - th1_0[i]
    for i in range(len(l1)):
        l1[i] = l1_0[i] - (dtheta1[i] * r)
    return l1

def grab_q(l1, s, d):
    """
    Parameters
    ---------------------------------
    l1: mod1 cable lengths
    s: arc length of module at neutral state
    d: circumradius of hexagon plate

    Returns
    ----------------------------------
    The generalized set of coordinates q in col vector form.
    q = [dx, dy, dL]
    """
 
    phi1 = math.atan2(((math.sqrt(3)/3) * (l1[1] + l1[0] - 2 * l1[2])),(l1[0] - l1[1]))
    k1 = 2 * math.sqrt(l1[2]**2 + l1[0]**2 + l1[1]**2 - (l1[2]*l1[0]) - (l1[0] * l1[1]) - (l1[2]*l1[1]))/(d* (l1[2] + l1[0] + l1[1]))

    s_curr = (l1[2] + l1[0] + l1[1])/3
    # can safely assume this is never poisitive because module doesn't expand past resting state length
    kx = (k1 * cos(phi1)/4) * s_curr 
    ky = (k1 * sin(phi1)/4) * s_curr
    theta5 = s_curr - s
    deltax = k1 * d * cos(phi1)
    deltay = k1 * d * sin(phi1)
    return np.array([deltax, deltay, theta5]).reshape(-1,1)


# grab_q([0.075, 0.025, 0.075], 0.075, 0.043)

