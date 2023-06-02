#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt, sin, cos
from mass import mass

def turtle_controller(q,dq,qd,dqd,ddqd, Kp, KD):

    M, C = mass(q, dq)
    tau = C.dot(dq) + M.dot(ddqd) + Kp.dot((qd-q)) + KD*(dqd - dq)

    return tau