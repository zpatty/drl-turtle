#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel"
sys.path.append(submodule)
import numpy as np
from math import sqrt, sin, cos
from mass import mass

def turtle_controller(q,dq,qd,dqd,ddqd, Kp, KD):

    M, C, G = mass(q, dq)
    tau = C.dot(dq) + M.dot(ddqd) + Kp.dot((qd-q)) + KD*(dqd - dq) + G.reshape(-1,1)

    # print(f"[DEBUG] G: {G}\n")
    # tau[1] = -tau[1]
    
    return tau*150