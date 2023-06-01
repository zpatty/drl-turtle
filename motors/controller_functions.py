#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import log, exp
import numpy.matlib
from math import sqrt, sin, cos
from scipy.linalg import block_diag
def vee(g):
    # TODO: controller doesn't use this function? --> test implementation later
    R = g[:3,:3]
    v = g[:3,3]
    twist = np.zeros((3, 4))
    # twist = [v;R[3,2];R[1,3];R[2,1]]
    twist[:3, :3] = v
    twist[:3, 3:] = R[3,2]
    twist[:3, 3:] = R[1,3]
    twist[:3, 3:] = R[2,1]
    return twist
    
def spatial_cross(V):

    v = np.array(V[:3]).reshape(-1,1)
    w = np.array(V[3:]).reshape(-1,1)
    Vx = np.zeros((6,6))
    skw = skew(w)
    Vx[:3,:3] = skw
    Vx[3:,3:] = skw
    Vx[:3,3:] = skew(v)
    return Vx

def skew(x):
    '''
    Function that takes in a vector and creates the skew matrix as defined in wikipedia 
    '''
    X = np.array([[0,  -x[2][0],  x[1][0]],
                    [x[2][0],  0,  -x[0][0]],
                    [-x[1][0],  x[0][0],  0]])
    return X

def adj_calc(g, inv):
    '''
    This function calculates the adjoint given simply the transform between joints
    '''
    # convert transform to rotation and translation
    R = g[:3,:3]
    p = np.array(g[:3, 3]).reshape(-1,1)
    # get skew symmetric matrix of translation
    p_hat = skew(p)
    adj = np.zeros((6, 6))
    if inv == 0:
        # package into adjoint
        # adj = [R p_hat*R; zeros(3) R]
        adj[:3, :3] = R
        adj[:3, 3:] = p_hat@R
        adj[3:, :3] = np.zeros((3, 1))
        adj[3:, 3:] = R
    else:
        # adj = [R.' -R.'*p_hat; zeros(3) R.']
        adj[:3, :3] = R.T
        adj[:3, 3:] = -R.T@p_hat
        adj[3:, :3] = np.zeros((3, 1))
        adj[3:, 3:] = R.T
    return adj

def repmat(arr, m, n):
    mat = np.array(np.matlib.repmat(arr, m, n))
    return mat
def blkdiag(a):
    return block_diag(a[0],a[1], a[2], a[3], a[4])
