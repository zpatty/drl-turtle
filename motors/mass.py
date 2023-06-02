#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt, sin, cos
def mass(q, dq):
    q2 = q[1][0]
    q3 = q[2][0]

    dq1 = q[0][0]
    dq2 = q[1][0]
    dq3 = q[2][0]

    m = 0.2
    l = 0.065
    w = 0.044
    h = 0.034
    lf = 0.19
    wf = 0.09
    hf = 0.025
    lm = 0.06
    m1 = m
    m2 = m
    m3 = m
    m4 = 0.5
    I11 = 1/12*m*(l**2 + h**2)
    I12 = 1/12*m*(w**2 + h**2)
    I13 = 1/12*m*(w**2 + l**2)
    I21 = 1/12*m*(w**2 + h**2)
    I22 = 1/12*m*(w**2 + l**2)
    I23 = 1/12*m*(h**2 + l**2)
    I31 = 1/12*m*(h**2 + w**2)
    I32 = 1/12*m*(h**2 + l**2)
    I33 = 1/12*m*(w**2 + l**2)
    I41 = 1/12*m*(lf**2 + hf**2)
    I42 = 1/12*m*(wf**2 + hf**2)
    I43 = 1/12*m*(wf**2 + lf**2)
    l1 = 0.06
    l2 = 0.065
    lo1 = lm/2 - 0.015
    lo2 = lm/2 - 0.015
    lo3 = 0.025
    lo4 = lm/2 - 0.015
    
    
    
    
    M = np.array([[I21 + I32 + I42 + l2**2*m2 + l2**2*m3 + l2**2*m4 + (lf**2*m4)/4 + lo2**2*m3 + lo2**2*m4 + lo3**2*m3 + lo3**2*m4 + lo4**2*m4 + I31*cos(q2)**2 - I32*cos(q2)**2 + (I41*cos(q2)**2)/2 - I42*cos(q2)**2 + (I43*cos(q2)**2)/2 + (I41*cos(2*q3)*cos(q2)**2)/2 - (I43*cos(2*q3)*cos(q2)**2)/2 + lf*lo2*m4 + lf*lo4*m4 + 2*lo2*lo4*m4 - (lf**2*m4*cos(q2)**2)/4 - lo2**2*m3*cos(q2)**2 - lo2**2*m4*cos(q2)**2 - lo4**2*m4*cos(q2)**2 + (m4*wf**2*cos(q2)**2)/4 + l2*m4*wf*cos(q2) - l2*lf*m4*sin(q2) - 2*l2*lo2*m3*sin(q2) - 2*l2*lo2*m4*sin(q2) - 2*l2*lo4*m4*sin(q2) - lf*lo2*m4*cos(q2)**2 - lf*lo4*m4*cos(q2)**2 - 2*lo2*lo4*m4*cos(q2)**2 - (lf*m4*wf*sin(2*q2))/4 - (lo2*m4*wf*sin(2*q2))/2 - (lo4*m4*wf*sin(2*q2))/2, (I41*sin(2*q3)*cos(q2))/2 - (I43*sin(2*q3)*cos(q2))/2 + (lf*lo3*m4*cos(q2))/2 + lo2*lo3*m3*cos(q2) + lo2*lo3*m4*cos(q2) + lo3*lo4*m4*cos(q2) + (lo3*m4*wf*sin(q2))/2, I42*sin(q2)],
                [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (I41*sin(2*q3)*cos(q2))/2 - (I43*sin(2*q3)*cos(q2))/2 + (lf*lo3*m4*cos(q2))/2 + lo2*lo3*m3*cos(q2) + lo2*lo3*m4*cos(q2) + lo3*lo4*m4*cos(q2) + (lo3*m4*wf*sin(q2))/2,      I33 + I41/2 + I43/2 + (lf**2*m4)/4 + lo2**2*m3 + lo2**2*m4 + lo4**2*m4 + (m4*wf**2)/4 - (I41*cos(2*q3))/2 + (I43*cos(2*q3))/2 + lf*lo2*m4 + lf*lo4*m4 + 2*lo2*lo4*m4,           0],
                [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I42*sin(q2),                                                                                                                                                                    0,         I42]])

    C = np.array([[- dq2*(2*I31*cos(q2)*sin(q2) - 2*I32*cos(q2)*sin(q2) + I41*cos(q2)*sin(q2) - 2*I42*cos(q2)*sin(q2) + I43*cos(q2)*sin(q2) - (lf**2*m4*cos(q2)*sin(q2))/2 - 2*lo2**2*m3*cos(q2)*sin(q2) - 2*lo2**2*m4*cos(q2)*sin(q2) - 2*lo4**2*m4*cos(q2)*sin(q2) + (m4*wf**2*cos(q2)*sin(q2))/2 + l2*lf*m4*cos(q2) + 2*l2*lo2*m3*cos(q2) + 2*l2*lo2*m4*cos(q2) + 2*l2*lo4*m4*cos(q2) + l2*m4*wf*sin(q2) + I41*cos(2*q3)*cos(q2)*sin(q2) - I43*cos(2*q3)*cos(q2)*sin(q2) + (lf*m4*wf*cos(2*q2))/2 + lo2*m4*wf*cos(2*q2) + lo4*m4*wf*cos(2*q2) - 2*lf*lo2*m4*cos(q2)*sin(q2) - 2*lf*lo4*m4*cos(q2)*sin(q2) - 4*lo2*lo4*m4*cos(q2)*sin(q2)) - dq3*(I41*sin(2*q3)*cos(q2)**2 - I43*sin(2*q3)*cos(q2)**2),
    dq3*(I42*cos(q2) + I41*cos(2*q3)*cos(q2) - I43*cos(2*q3)*cos(q2)) - dq1*(2*I31*cos(q2)*sin(q2) - 2*I32*cos(q2)*sin(q2) + I41*cos(q2)*sin(q2) - 2*I42*cos(q2)*sin(q2) + I43*cos(q2)*sin(q2) - (lf**2*m4*cos(q2)*sin(q2))/2 - 2*lo2**2*m3*cos(q2)*sin(q2) - 2*lo2**2*m4*cos(q2)*sin(q2) - 2*lo4**2*m4*cos(q2)*sin(q2) + (m4*wf**2*cos(q2)*sin(q2))/2 + l2*lf*m4*cos(q2) + 2*l2*lo2*m3*cos(q2) + 2*l2*lo2*m4*cos(q2) + 2*l2*lo4*m4*cos(q2) + l2*m4*wf*sin(q2) + I41*cos(2*q3)*cos(q2)*sin(q2) - I43*cos(2*q3)*cos(q2)*sin(q2) + (lf*m4*wf*cos(2*q2))/2 + lo2*m4*wf*cos(2*q2) + lo4*m4*wf*cos(2*q2) - 2*lf*lo2*m4*cos(q2)*sin(q2) - 2*lf*lo4*m4*cos(q2)*sin(q2) - 4*lo2*lo4*m4*cos(q2)*sin(q2)) - dq2*(I41*sin(2*q3)*sin(q2) - I43*sin(2*q3)*sin(q2) - lo3*m4*wf*cos(q2) + lf*lo3*m4*sin(q2) + 2*lo2*lo3*m3*sin(q2) + 2*lo2*lo3*m4*sin(q2) + 2*lo3*lo4*m4*sin(q2)), 
    dq2*(I42*cos(q2) + I41*cos(2*q3)*cos(q2) - I43*cos(2*q3)*cos(q2)) - dq1*(I41*sin(2*q3)*cos(q2)**2 - I43*sin(2*q3)*cos(q2)**2)],
    [dq1*(2*I31*cos(q2)*sin(q2) - 2*I32*cos(q2)*sin(q2) + I41*cos(q2)*sin(q2) - 2*I42*cos(q2)*sin(q2) + I43*cos(q2)*sin(q2) - (lf**2*m4*cos(q2)*sin(q2))/2 - 2*lo2**2*m3*cos(q2)*sin(q2) - 2*lo2**2*m4*cos(q2)*sin(q2) - 2*lo4**2*m4*cos(q2)*sin(q2) + (m4*wf**2*cos(q2)*sin(q2))/2 + l2*lf*m4*cos(q2) + 2*l2*lo2*m3*cos(q2) + 2*l2*lo2*m4*cos(q2) + 2*l2*lo4*m4*cos(q2) + l2*m4*wf*sin(q2) + I41*cos(2*q3)*cos(q2)*sin(q2) - I43*cos(2*q3)*cos(q2)*sin(q2) + (lf*m4*wf*cos(2*q2))/2 + lo2*m4*wf*cos(2*q2) + lo4*m4*wf*cos(2*q2) - 2*lf*lo2*m4*cos(q2)*sin(q2) - 2*lf*lo4*m4*cos(q2)*sin(q2) - 4*lo2*lo4*m4*cos(q2)*sin(q2)) - dq3*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2)),
    dq3*(I41*sin(2*q3) - I43*sin(2*q3)),
    dq2*(I41*sin(2*q3) - I43*sin(2*q3)) - dq1*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2))],
    [dq2*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2)) + dq1*(I41*sin(2*q3)*cos(q2)**2 - I43*sin(2*q3)*cos(q2)**2),
    dq1*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2)) - dq2*(I41*sin(2*q3) - I43*sin(2*q3)),
    0]])
    
    return M, C